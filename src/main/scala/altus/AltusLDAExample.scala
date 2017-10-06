package altus

// Workbench users: execute only the code between START and END blocks

import org.apache.spark.sql.SparkSession

object AltusLDAExample {


  // START Workbench ------------------------------

  case class Params(
      dataDir: String = "hdfs:///user/ds/gutenberg", // Remember to update this!
      sampleRate: Double = 0.1,
      kValues: String = "10,30",
      maxIter: Int = 10) {
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
    import org.apache.spark.sql.functions.{udf, substring, lit}

    // Parse raw text into lines, ignoring boilerplate header/footer
    val newlinesRegex = "\n+".r
    val headerEndRegex =
      """((START OF THE PROJECT GUTENBERG EBOOK
        |(\*+)(\*+)(\*+) START OF THE PROJECT GUTENBERG EBOOK
        |START OF THIS PROJECT GUTENBERG EBOOK
        |((\<+)(\<+))THIS ELECTRONIC VERSION OF THE COMPLETE WORKS OF WILLIAM
        |These original Project Gutenberg Etexts will be compiled into a file
        |computers we used then didn't have lower case at all.)+) *""".r
    val footerStartRegex =
      """((End of The Project Gutenberg EBook of )+) *""".r
    val languageRegex =
      """((Language: [a-zA-Z]+)+) *""".r
    val CIABookRegex = """Produced by Dr. Gregory B. Newby""".r


    val stripHeaderFooterUDF = udf { text: String =>
      val lines = newlinesRegex.split(text).map(_.trim).filter(_.nonEmpty)
      val headerEnd = lines.indexWhere(headerEndRegex.findFirstIn(_).isDefined)
      val footerStart = lines.indexWhere(footerStartRegex.findFirstIn(_).isDefined, headerEnd)
      lines.slice(if (headerEnd < 0) 0 else headerEnd + 1,
        if (footerStart < 0) lines.length else footerStart).mkString(" ")
    }
    def findLanguageUDF = udf { text: String =>
      val lines = newlinesRegex.split(text).map(_.trim).filter(_.nonEmpty)
      val start = lines.indexWhere(CIABookRegex.findFirstIn(_).isDefined)
      if (start < 0) {
        languageRegex.findFirstIn(text).mkString(" ").trim.toUpperCase
      } else {
        "LANGUAGE: ENGLISH"
      }
    }

    val getSubstrUDF = udf { x: String => x.substring(x.length-200) }
    val engBooksRegex = """(((\/+)(\d+)(\.+)txt)|((\/+)(\d+)(\-+)(0|8)(\.+)txt))""".r

    val findFilenameUDF = udf { x: String => engBooksRegex.findFirstIn(x) }
    val allTexts = spark.read.parquet(params.dataDir).
      withColumn("textStripped", stripHeaderFooterUDF($"text")).
      withColumn("language", findLanguageUDF($"text")).
      withColumn("startText", substring($"text", 1, 200)).
      withColumn("filename", findFilenameUDF($"path"))

    println(s"num of obs: ${allTexts.count()}")
    allTexts.select("path", "language", "startText", "filename").show(5, false)

    val allEngTexts = allTexts.filter($"filename" =!= "null" && ($"language" === "LANGUAGE: ENGLISH"
      || $"language" === "LANGUAGE: EN" || $"language" === "LANGUAGE: ENGLISHS" || $"language" === ""))

    println(s"num of obs after filtering: ${allEngTexts.count()}")
    allEngTexts.select("path", "language", "filename").show(90000, false)

    // Split each document into words
    val tokens = new RegexTokenizer().
      setGaps(false).
      setPattern("\\p{L}+").
      setInputCol("textStripped").
      setOutputCol("words").
      transform(allEngTexts)

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
      transform(tokens)
      //select("path", "tokens")

    // Sample a subset
    val sampleSubset = if (params.sampleRate < 1.0) {
      filteredTokens.sample(withReplacement = false, fraction = params.sampleRate, seed=123)
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
    println(s"Vocabulary: ${vocabModel.vocabulary.mkString(",")}")

    // Obtain a train/test split of featurized data, and cache
    val Array(train, test) = docTermFreqs.randomSplit(Array(0.9, 0.1), seed=123)
    train.cache()
    test.cache()
    println(s"Train size: ${train.count()}")
    println(s"Test size:  ${test.count()}")
    test.show()

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
      withColumn("topic", findIndexMax($"topicDistribution")).
      withColumn("split", lit("test"))

    println("Example topic assignments from test set")
    testScored.select("path", "startText", "topic", "topicDistribution").show(10, false)

    val trainScored = bestModel.
      transform(train).
      drop("features").
      withColumn("topic", findIndexMax($"topicDistribution")).
      withColumn("split", lit("train"))

    val scored = trainScored.union(testScored)
    scored.write.parquet("hdfs:///user/nisha/Data/scored_SampleLDAK10Iter20")

    // END Workbench ------------------------------


  }

}

