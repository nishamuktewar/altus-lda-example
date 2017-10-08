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
    import org.apache.spark.ml.feature.{CountVectorizer, RegexTokenizer, StopWordsRemover}
    import org.apache.spark.ml.linalg.Vector
    import org.apache.spark.sql.functions.udf

    // Parse raw text into lines, ignoring boilerplate header/footer
    val newlinesRegex = "\n+".r
    val headerEndRegex = "(START.+PROJECT GUTENBERG|SMALL PRINT|TIME OR FOR MEMBERSHIP)".r
    val footerStartRegex = "(?i)(END.+PROJECT GUTENBERG.+E(BOOK|TEXT)|<<THIS ELECTRONIC VERSION OF)".r

    val allTexts = spark.read.parquet(params.dataDir).as[(String,String)].flatMap { case (path, text) =>
      val lines = newlinesRegex.split(text).map(_.trim).filter(_.nonEmpty)
      // Keep only docs explicitly marked as English
      if (lines.exists(_.contains("Language: English"))) {
        // Try to find lines that are boilerplate header/footer and remove them
        val headerEnd = lines.indexWhere(headerEndRegex.findFirstIn(_).isDefined)
        val footerStart = lines.indexWhere(footerStartRegex.findFirstIn(_).isDefined, headerEnd)
        val textLines = lines.slice(
            if (headerEnd < 0) 0 else headerEnd + 1,
            if (footerStart < 0) lines.length else footerStart).
          mkString(" ")
        Some((path, textLines))
      } else {
        None
      }
    }.toDF("path", "text")

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
      select("path", "tokens")

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
      setOutputCol("features")
    val vocabModel = countVectorizer.fit(sampleSubset)
    val docTermFreqs = vocabModel.transform(sampleSubset)

    docTermFreqs.cache()

    // Obtain a train/test split of featurized data, and cache
    val Array(train, test) = docTermFreqs.randomSplit(Array(0.9, 0.1), seed = params.rngSeed)
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
    val bestModel = modelsWithPerplexity.minBy(_._1)._2

    train.unpersist()
    test.unpersist()

    println("Best model top topics (by term weight):")
    val topicIndices =
      bestModel.describeTopics(10).
      select("termIndices", "termWeights").
      as[(Array[Int], Array[Double])].
      collect()

    topicIndices.zipWithIndex.foreach { case ((terms, termWeights), i) =>
      println(s"TOPIC $i")
      terms.zip(termWeights).foreach { case (term, weight) =>
        println(s"${vocabModel.vocabulary(term)}\t$weight")
      }
      println()
    }

    // Lookup the topic with highest probability
    val findIndexMax = udf { x: Vector => x.argmax }
    val scored = bestModel.
      transform(docTermFreqs).
      select("path", "topicDistribution").
      withColumn("topic", findIndexMax($"topicDistribution"))

    println("Example topic assignments:")
    scored.show(10)

    if (params.outputPath.nonEmpty) {
      scored.write.parquet(params.outputPath)
    }

    // END Workbench ------------------------------


  }

}

