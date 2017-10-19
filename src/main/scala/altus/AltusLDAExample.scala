package altus

// Workbench users: execute only the code between START and END blocks

import org.apache.spark.sql.SparkSession

object AltusLDAExample {


  // START Workbench ------------------------------

  case class Params(
      dataDir: String = "hdfs:///user/ds/gutenberg", // Remember to update this!
      outputPath: String = "", // Remember to update this!
      sampleRate: Double = 0.1,
      kValues: String = "10,30",
      maxIter: Int = 10,
      rngSeed: Int = 123) {
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
    import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover, IDF}
    import org.apache.spark.ml.linalg.Vector
    import org.apache.spark.sql.functions.udf

    // Parse raw text into lines, ignoring boilerplate header/footer
    val newlinesRegex = "\n+".r
    val headerEndRegex = "(START.+PROJECT GUTENBERG|SMALL PRINT|TIME OR FOR MEMBERSHIP)".r
    val footerStartRegex = "(?i)(END.+PROJECT GUTENBERG.+E(BOOK|TEXT)|<<THIS ELECTRONIC VERSION OF)".r

    val allTexts = spark.read.parquet(params.dataDir).as[(String,String)].flatMap { case (path, text) =>
      val lines = newlinesRegex.split(text).map(_.trim).filter(_.nonEmpty)
      val title = lines.find(_.startsWith("Title:")).map(_.substring("Title:".length).trim).orNull
      // Keep only docs explicitly marked as English
      if (lines.exists(_.contains("Language: English"))) {
        // Try to find lines that are boilerplate header/footer and remove them
        val headerEnd = lines.indexWhere(headerEndRegex.findFirstIn(_).isDefined)
        val footerStart = lines.indexWhere(footerStartRegex.findFirstIn(_).isDefined, headerEnd)
        val textLines = lines.slice(
            if (headerEnd < 0) 0 else headerEnd + 1,
            if (footerStart < 0) lines.length else footerStart).
          mkString(" ")
        Some((path, textLines, title))
      } else {
        None
      }
    }.toDF("path", "text", "title")

    // Split each document into words
    val tokens = new RegexTokenizer().
      setGaps(false).
      setPattern("\\p{L}+").
      setInputCol("text").
      setOutputCol("words").
      transform(allTexts)

    // Filter out stopwords
    val stopwordsFile = new File("src/main/resources/stopwords.txt")
    val stopwordsStream =
      if (stopwordsFile.exists()) {
        // Try reading local file; working locally
        Source.fromFile(stopwordsFile)
      } else {
        // Try reading from classpath; deployed app
        Source.fromInputStream(
          this.getClass.getClassLoader.getResourceAsStream(stopwordsFile.getName))
      }
    val stopwords = stopwordsStream.getLines().toArray
    stopwordsStream.close()

    val filteredTokens = new StopWordsRemover().
      setStopWords(stopwords).
      setCaseSensitive(false).
      setInputCol("words").
      setOutputCol("tokens").
      transform(tokens).
      select("path", "title", "tokens")

    // Sample a subset
    val sampleSubset = if (params.sampleRate < 1.0) {
      filteredTokens.sample(withReplacement = false, fraction = params.sampleRate, seed = params.rngSeed)
    } else {
      filteredTokens
    }

    sampleSubset.cache()

    // Learn the vocabulary of the whole test set
    val countVectorizer = new CountVectorizer().
      setVocabSize(65536).
      setInputCol("tokens").
      setOutputCol("rawFeatures")
    val vocabModel = countVectorizer.fit(sampleSubset)
    val docTermFreqs = vocabModel.transform(sampleSubset)

    val idf = new IDF().
      setInputCol("rawFeatures").
      setOutputCol("features")
    val idfModel = idf.fit(docTermFreqs)
    val modelingData = idfModel.transform(docTermFreqs).drop("rawFeatures")

    // Obtain a train/test split of featurized data, and cache
    val Array(train, test) = modelingData.randomSplit(Array(0.9, 0.1), seed = params.rngSeed)
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
          setSeed(params.rngSeed).
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

    val (_, bestModel) = modelsWithPerplexity.minBy(_._1)

    train.unpersist()
    test.unpersist()

    val vocabulary = vocabModel.vocabulary

    val topTopicTerms =
      bestModel.describeTopics(10).
        select("topic", "termIndices").
        as[(Int, Array[Int])].
        collect().toMap.
        mapValues(_.map(vocabulary))

    val scored = bestModel.
      transform(modelingData).
      select("path", "title", "topicDistribution")
    scored.cache()

    for (topic <- 0 until bestModel.getK) {
      println(s"TOPIC $topic")
      println(topTopicTerms(topic).mkString(", "))
      val topicConcUDF = udf { x: Vector => x(topic) }
      scored.
        withColumn("topicConc", topicConcUDF($"topicDistribution")).
        orderBy($"topicConc".desc).
        select("title").
        show(10, false)
      println()
    }

    if (params.outputPath.nonEmpty) {
      scored.write.parquet(params.outputPath)
    }

    scored.unpersist()

    // END Workbench ------------------------------

  }

}

