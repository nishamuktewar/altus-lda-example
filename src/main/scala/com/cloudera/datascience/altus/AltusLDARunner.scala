package com.cloudera.datascience.altus

import com.cloudera.datascience.altus.AltusLDAExample.Params
import org.apache.spark.sql.SparkSession
import scopt.OptionParser

object AltusLDARunner {

  def main(args: Array[String]): Unit = {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("AltusLDAExample") {
      opt[String]("dataDir")
        .text(s"HDFS location of the text files" +
          s"  default: ${defaultParams.dataDir}")
        .action((x, c) => c.copy(dataDir = x))
      opt[Double]("sampleRate")
        .text("fraction of data to use" +
          s"  default: ${defaultParams.sampleRate}")
        .action((x, c) => c.copy(sampleRate = x))
      opt[String]("kValues")
        .text(s"the number of topics to try (comma-separated list)" +
          s"  default: ${defaultParams.kValues}")
        .action((x, c) => c.copy(kValues = x))
      opt[Int]("maxIter")
        .text("the number of iterations on the training data" +
          s"  default: ${defaultParams.maxIter}")
        .action((x, c) => c.copy(maxIter = x))
    }

    val spark = SparkSession.builder().appName("AltusLDAExample").getOrCreate()

    parser.parse(args, defaultParams) match {
      case Some(params) =>
        AltusLDAExample.params = params
        AltusLDAExample.run(spark)
      case _ =>
        sys.error("Error parsing args")
    }
  }

}
