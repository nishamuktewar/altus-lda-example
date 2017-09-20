/**
  * Created by nisha on 9/18/17.
  */

package com.cloudera.datascience

import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}
import org.apache.spark.ml.clustering.{LDA, LDAModel}
import scopt.OptionParser

object LDAExample {
  val gutenbergBase = "hdfs:///user/sowen/DataSets/gutenberg"
  val newlinesRegex = "\n+".r
  val headerEndRegex = """\*\*\*.*START.+PROJECT GUTENBERG.*\*\*\*""".r
  val footerStartRegex = """\*\*\*.*END.+PROJECT GUTENBERG.*\*\*\*""".r

  case class Params(
                     dataDir: String = gutenbergBase,
                     stopwordFile: String = "",
                     saveModelDir: String = ""
                   )

  def main(args: Array[String]): Unit = {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("LDAExample") {
      head("LDAExample: an example LDA app for gutenberg data.")
      opt[String]("dataDir")
        .text(s"hdfs location of the text files" +
          s"  default: ${defaultParams.dataDir}")
        .action((x, c) => c.copy(dataDir = x))
      opt[String]("stopwordFile")
        .text(s"filepath for a list of stopwords. Note: This must fit on a single machine." +
          s"  default: ${defaultParams.stopwordFile}")
        .action((x, c) => c.copy(stopwordFile = x))
      opt[String]("saveModelDir")
        .text(s"hdfs location to save the LDA model" +
          s"  default: ${defaultParams.saveModelDir}")
        .action((x, c) => c.copy(saveModelDir = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => preprocess(params)
      case _ => sys.exit(1)
    }
  }

  def preprocess(params: Params): Unit = {
    //Start a Spark session
    val spark = SparkSession.builder().appName("LDAExample: an example LDA app for gutenberg data.").getOrCreate()

    import spark.sqlContext.implicits._

    val allTexts = spark.sparkContext.union(
      spark.sparkContext.wholeTextFiles(s"${params.dataDir}/*/*/*.txt", 10),
      spark.sparkContext.wholeTextFiles(s"${params.dataDir}/*/*/*/*.txt", 100),
      spark.sparkContext.wholeTextFiles(s"${params.dataDir}/*/*/*/*/*.txt", 1000)
    ).values.map(stripHeaderFooter).toDF("text")

    println(s"# obs: ${allTexts.count()}")

    val allTextsDF = allTexts.withColumn("docId", monotonically_increasing_id())
    allTextsDF.show(5)

    // Split each document into words
    val tokens = new RegexTokenizer()
      .setGaps(false)
      .setPattern("\\p{L}+")
      .setInputCol("text")
      .setOutputCol("words")
      .transform(allTextsDF)

    tokens.show(5)

    // Filter out stopwords
    val stopwords: Array[String] = spark.read.text(params.stopwordFile).
      rdd.map(r => r.getString(0)).collect()
    println(s"stopwords: ${stopwords.mkString(",")}")

    val filteredTokens = new StopWordsRemover()
      .setStopWords(stopwords)
      .setCaseSensitive(false)
      .setInputCol("words")
      .setOutputCol("tokens")
      .transform(tokens)
    filteredTokens.show(5)

    // 10% random sample
    val modelingData = filteredTokens.sample(withReplacement = false, fraction = 0.1, seed=123)

    // Split data into 70%-30%
    val Array(train, test) = modelingData.randomSplit(Array(0.7, 0.3), seed=123)
    println(s"Number of obs in train ${train.count()}")
    println(s"Number of obs in test ${test.count()}")

    val vocabSize = 50000
    val countVectorizer = new CountVectorizer()
      .setVocabSize(vocabSize)
      .setInputCol("tokens")
      .setOutputCol("features")

    val vocabModel = countVectorizer.fit(train)
    val docTermFreqs = vocabModel.transform(train)
    val vocabulary = vocabModel.vocabulary
    println(s"vocab length: ${vocabulary.length}")

    val vocabModelTest = countVectorizer.fit(test)
    val docTermFreqsTest = vocabModelTest.transform(test)
    val vocabularyTest = vocabModelTest.vocabulary

    val startTime = System.nanoTime()
    val ldaModels = fitLDA(df = docTermFreqs, kValues = Seq(10), maxIter = Seq(100), subSampleRate = 0.05,
      optDocConcentration = true, docTermFreqsTest, saveModelDir = params.saveModelDir)
    val elapsed = (System.nanoTime() - startTime) / 1e9
    println(s"Finished training LDA model.  Summary:")
    println(s"Training time (sec)\t$elapsed")
    println(s"==========")
  }

  def stripHeaderFooter(text: String): String = {
    val lines = newlinesRegex.split(text).map(_.trim).filter(_.nonEmpty)
    val headerEnd = lines.indexWhere(headerEndRegex.findFirstIn(_).isDefined)
    val footerStart =
      lines.indexWhere(footerStartRegex.findFirstIn(_).isDefined, headerEnd)
    lines.slice(if (headerEnd < 0) 0 else headerEnd + 1,
      if (footerStart < 0) lines.length else footerStart)
      .mkString(" ")
  }

  /*
  Creates a sequence of LDA models (and saves them) based on:
  df - input dataframe
  kValues - list of num of topics
  maxIter - list of num of iterations
  subSampleRate - the sampling rate for minibatch sizes
  optDocConcentration - indicates whether the dirichlet param will be optimized for the training
  */

  def fitLDA(df: DataFrame, kValues: Seq[Int], maxIter: Seq[Int], subSampleRate: Double, optDocConcentration: Boolean,
             testDF: DataFrame, saveModelDir: String): Seq[Seq[LDAModel]] = {
    kValues.map { k =>
      maxIter.map { m =>
        println(s"Fitting LDA for parameters K = $k and maxIter = $m")
        val ldaModel = new LDA()
          .setMaxIter(m)
          .setOptimizer("online")
          .setSubsamplingRate(subSampleRate)
          .setOptimizeDocConcentration(optDocConcentration)
          .setSeed(123)
          .setK(k)
          .setFeaturesCol("features")
          .fit(df)
        println(s"Perplexity on test set: ${ldaModel.logPerplexity(testDF)}")
        ldaModel.save(s"${saveModelDir}LDAK${k}Iter${m}SampleRate${subSampleRate}")
        ldaModel
      }
    }
  }
}

