package com.cloudera.datascience

import scopt.OptionParser
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.linalg.Vector

import scala.io.Source

object LDAExample {

  case class Params(
      dataDir: String = "hdfs:///user/sowen/DataSets/gutenberg",
      stopwordFile: String = "stopwords.txt",
      saveModelDir: String = "",
      sampleRate: Double = 0.1,
      kValues: String = "10,30,100",
      maxIter: Int = 10) {
    def kValuesList: Seq[Int] = kValues.split(",").map(_.toInt)
  }

  def main(args: Array[String]): Unit = {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("AltusLDAExample") {
      head("AltusLDAExample:")
      opt[String]("dataDir")
        .text(s"HDFS location of the text files" +
          s"  default: ${defaultParams.dataDir}")
        .action((x, c) => c.copy(dataDir = x))
      opt[String]("stopwordFile")
        .text(s"local file path for a list of stopwords." +
          s"  default: ${defaultParams.stopwordFile}")
        .action((x, c) => c.copy(stopwordFile = x))
      opt[String]("saveModelDir")
        .text(s"HDFS location to save the LDA model" +
          s"  default: ${defaultParams.saveModelDir}")
        .action((x, c) => c.copy(saveModelDir = x))
      opt[Double]("sampleRate")
        .text("fraction of data to use" +
          s"  default: ${defaultParams.sampleRate}")
        .action((x, c) => c.copy(sampleRate = x))
      opt[String]("kValues")
        .text(s"the number of topics to try (comma-separated list)" +
          s"  default: ${defaultParams.kValues}")
        .action((x, c) => c.copy(kValues = x))
      opt[Int]("maxIter")
        .text("the number of iterations on the training data. " +
          "Note: kValue * maxIter and subSampleRate = 0.05 would " +
          "determine the number of passes over the entire data" +
          s"  default: ${defaultParams.maxIter}")
        .action((x, c) => c.copy(maxIter = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => runLDA(params)
      case _ => println("Error parsing args")
    }
  }

  private def runLDA(params: Params): Unit = {

    val spark = SparkSession.builder().appName("AltusLDAExample").getOrCreate()

    import spark.implicits._

    val newlinesRegex = "\n+".r
    val headerEndRegex = """\*\*\*.*START.+PROJECT GUTENBERG.*\*\*\*""".r
    val footerStartRegex = """\*\*\*.*END.+PROJECT GUTENBERG.*\*\*\*""".r

    def stripHeaderFooter(text: String): String = {
      val lines = newlinesRegex.split(text).map(_.trim).filter(_.nonEmpty)
      val headerEnd = lines.indexWhere(headerEndRegex.findFirstIn(_).isDefined)
      val footerStart = lines.indexWhere(footerStartRegex.findFirstIn(_).isDefined, headerEnd)
      lines.slice(if (headerEnd < 0) 0 else headerEnd + 1,
        if (footerStart < 0) lines.length else footerStart).mkString(" ")
    }

    val allTexts = spark.sparkContext.union(
      spark.sparkContext.wholeTextFiles(s"${params.dataDir}/*/*/*.txt", 2),
      spark.sparkContext.wholeTextFiles(s"${params.dataDir}/*/*/*/*.txt", 20),
      spark.sparkContext.wholeTextFiles(s"${params.dataDir}/*/*/*/*/*.txt", 120)
    ).values.map(stripHeaderFooter).toDF("text")

    // Split each document into words
    val tokens = new RegexTokenizer().
      setGaps(false).
      setPattern("\\p{L}+").
      setInputCol("text").
      setOutputCol("words").
      transform(allTexts)

    // Filter out stopwords
    val stopwords = Source.fromFile(params.stopwordFile).getLines().toArray

    val filteredTokens = new StopWordsRemover().
      setStopWords(stopwords).
      setCaseSensitive(false).
      setInputCol("words").
      setOutputCol("tokens").
      transform(tokens)

    val modelingData = filteredTokens.
      select("tokens").
      sample(withReplacement = false, fraction = params.sampleRate, seed=123)

    modelingData.cache()

    val countVectorizer = new CountVectorizer().
      setVocabSize(100000).
      setInputCol("tokens").
      setOutputCol("features")
    val vocabModel = countVectorizer.fit(modelingData)
    val vocabulary = vocabModel.vocabulary

    val Array(train, test) = modelingData.randomSplit(Array(0.9, 0.1), seed=123)
    val docTermFreqs = vocabModel.transform(train)
    val docTermFreqsTest = vocabModel.transform(test)

    docTermFreqs.cache()
    docTermFreqsTest.cache()

    println(s"Train size: ${docTermFreqs.count()}")
    println(s"Test size:  ${docTermFreqsTest.count()}")

    modelingData.unpersist()

    val ldaModels =
      params.kValuesList.par.map { k =>
        println(s"Fitting LDA for parameters k = $k")
        val ldaModel = new LDA().
          setMaxIter(params.maxIter).
          setOptimizer("online").
          setSubsamplingRate(1.0).
          setOptimizeDocConcentration(true).
          setSeed(123).
          setK(k).
          setFeaturesCol("features").
          fit(docTermFreqs)
        println(s"Perplexity on test set: ${ldaModel.logPerplexity(docTermFreqsTest)}")
        //ldaModel.save(s"${saveModelDir}LDAK${k}Iter${m}SampleRate${subSampleRate}")
        ldaModel
      }.toList

    val modelsWithPerplexity = ldaModels.
      map(model => (model.logPerplexity(docTermFreqsTest), model))

    modelsWithPerplexity.foreach { case (perplexity, model) =>
      println(s"k = ${model.getK} : $perplexity")
    }
    println()

    val bestModel = modelsWithPerplexity.minBy(_._1)._2

    println("Best model top topics (by term weight):")
    val topicIndices =
      bestModel.describeTopics(maxTermsPerTopic = 10).
      select("termIndices", "termWeights").
      as[(Array[Int], Array[Double])].
      collect()

    topicIndices.zipWithIndex.foreach { case ((terms, termWeights), i) =>
      println(s"TOPIC $i")
      terms.zip(termWeights).foreach { case (term, weight) =>
        println(s"${vocabulary(term)}\t$weight")
      }
      println()
    }

    // Lookup the topic with highest probability
    val findIndexMax = udf { x: Vector => x.argmax }
    val trainScored = bestModel.
      transform(docTermFreqs).
      drop("features").
      withColumn("topic", findIndexMax($"topicDistribution"))

    val testScored = bestModel.
      transform(docTermFreqsTest).
      drop("features").
      withColumn("topic", findIndexMax($"topicDistribution"))

    trainScored.show(10)
    testScored.show(10)
  }

}

