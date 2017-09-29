package com.cloudera.datascience.altus

import org.apache.spark.sql.SparkSession

// Workbench users: execute only the code between START and END blocks

object AltusLDAExample {


  // START Workbench ------------------------------

  case class Params(
      dataDir: String = "hdfs:///user/sowen/DataSets/gutenberg",
      sampleRate: Double = 0.1,
      kValues: String = "10,30,100",
      maxIter: Int = 20) {
    def kValuesList: Seq[Int] = kValues.split(",").map(_.trim.toInt)
  }

  var params = Params()

  // END Workbench ------------------------------


  private[altus] def run(spark: SparkSession): Unit = {

    import spark.implicits._


    // START Workbench ------------------------------

    import java.io.File
    import scala.io.Source
    import org.apache.spark.ml.clustering.LDA
    import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}
    import org.apache.spark.ml.linalg.Vector
    import org.apache.spark.sql.functions.udf

    // Parse raw text into lines, ignoring boilerplate header/footer
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

    // Must glob 3 times to get all files at all levels of the Gutenberg archive.
    // Default partitioning would leave very small and large partitions as few are at
    // the higher levels and many at lower levels.
    // Explicitly set partitions in order to roughly reflect the number of files
    // at each level.
    val basePartitions = 2
    val allTexts = spark.sparkContext.union(
      spark.sparkContext.wholeTextFiles(s"${params.dataDir}/*/*/*.txt", basePartitions),
      spark.sparkContext.wholeTextFiles(s"${params.dataDir}/*/*/*/*.txt", 10 * basePartitions),
      spark.sparkContext.wholeTextFiles(s"${params.dataDir}/*/*/*/*/*.txt", 63 * basePartitions)
    ).mapValues(stripHeaderFooter).toDF("path", "text")

    // Split each document into words
    val tokens = new RegexTokenizer().
      setGaps(false).
      setPattern("\\p{L}+").
      setInputCol("text").
      setOutputCol("words").
      transform(allTexts)

    // Filter out stopwords
    val stopwordsPath = "src/main/resources/stopwords.txt"
    val stopwordsFile = new File(stopwordsPath)
    val stopwordsStream =
      if (stopwordsFile.exists()) {
        // Try reading local file; working locally
        Source.fromFile(stopwordsFile)
      } else {
        // Try reading from classpath; deployed app
        Source.fromInputStream(
          this.getClass.getClassLoader.getResourceAsStream(stopwordsPath))
      }
    val stopwords = stopwordsStream.getLines().toArray
    stopwordsStream.close()

    val filteredTokens = new StopWordsRemover().
      setStopWords(stopwords).
      setCaseSensitive(false).
      setInputCol("words").
      setOutputCol("tokens").
      transform(tokens).
      select("path", "tokens")

    // Sample a subset
    val sampleSubset = if(params.sampleRate < 1.0) {
      filteredTokens.sample(withReplacement = false, fraction = params.sampleRate, seed=123)
    } else {
      filteredTokens
    }

    sampleSubset.cache()

    // Learn the vocabulary of the whole test set
    val countVectorizer = new CountVectorizer().
      setVocabSize(100000).
      setInputCol("tokens").
      setOutputCol("features")
    val vocabModel = countVectorizer.fit(sampleSubset)
    val docTermFreqs = vocabModel.transform(sampleSubset)

    // Obtain a train/test split of featurized data, and cache
    val Array(train, test) = docTermFreqs.randomSplit(Array(0.9, 0.1), seed=123)
    train.cache()
    test.cache()
    println(s"Train size: ${train.count()}")
    println(s"Test size:  ${test.count()}")

    sampleSubset.unpersist()

    // Fit a model for each value of k, in parallel
    val models =
      params.kValuesList.par.map { k =>
        println(s"Fitting LDA for k = $k")
        val model = new LDA().
          setMaxIter(params.maxIter).
          setOptimizer("online").
          setSeed(123).
          setK(k).
          setFeaturesCol("features").
          fit(train)
        println(s"Perplexity for k = $k on test set: ${model.logPerplexity(test)}")
        model
      }.toList

    // Print models and pick best by perplexity
    val modelsWithPerplexity = models.
      map(model => (model.logPerplexity(test), model))
    modelsWithPerplexity.foreach { case (perplexity, model) =>
      println(s"k = ${model.getK} : $perplexity")
    }
    println()
    val bestModel = modelsWithPerplexity.minBy(_._1)._2

    println("Best model top topics (by term weight):")
    val topicIndices =
      bestModel.describeTopics(10).
      select("termIndices", "termWeights").
      as[(Array[Int], Array[Double])].
      collect()

    val vocabulary = vocabModel.vocabulary
    topicIndices.zipWithIndex.foreach { case ((terms, termWeights), i) =>
      println(s"TOPIC $i")
      terms.zip(termWeights).foreach { case (term, weight) =>
        println(s"${vocabulary(term)}\t$weight")
      }
      println()
    }

    // Lookup the topic with highest probability
    val findIndexMax = udf { x: Vector => x.argmax }
    val testScored = bestModel.
      transform(test).
      drop("features").
      withColumn("topic", findIndexMax($"topicDistribution"))
    println("Example topic assignments from test set")
    testScored.show(10)

    // END Workbench ------------------------------


  }

}

