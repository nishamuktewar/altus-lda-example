# Clustering Books with LDA, Cloudera Data Science Workbench and Cloudera Altus

This brief example demonstrates clustering text documents using 
[Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) (LDA). In particular
this example uses a large corpus of public-domain texts from [Project Gutenberg](https://www.gutenberg.org/).

This shows how a data scientist might use the 
[Cloudera Data Science Workbench](https://www.cloudera.com/products/data-science-and-engineering/data-science-workbench.html) 
to interactively explore data and build models on a small cluster, and then leverage a much larger transient cluster
provisioned with [Cloudera Altus](https://www.cloudera.com/products/altus.html) to select and build the best, 
final model with additional computing power from Amazon Web Services (AWS).

## Exploring with the Workbench

A data scientist might first try experimenting in the Workbench, and refine a model building and selection process
there, with the help of a small CDH cluster. Once the modeling job is ready, it can be run at larger scale
with Altus in the cloud to build a better model.

### Get Data

This example requires the complete Project Gutenberg archive. While the public domain text can be copied
from the project mirrors, it's very large. A compressed archive with only the text files can be downloaded,
decompressed, and uploaded to a directory like `/user/ds/gutenberg` on HDFS as follows:

```bash
curl https://storage.googleapis.com/altus-cdsw-lda-example/gutenberg-20170929.tgz | tar xz
hdfs dfs -put gutenberg /user/ds/
```

### Checkout and Run Code in the Workbench

Start by creating a new project in the Workbench project based on the Git repo at
https://github.com/nisha/altus-lda-example and open it in the Workbench.

The file [spark-defaults.conf](blob/master/spark-defaults.conf) contains default resource sizing
for a moderately powerful 5-node cluster. This sizing can be increased or reduced to match your
cluster.

Run a Scala session with 4 CPUs and 8GB of RAM. (More, or less, depending on your driver resource size.)

[AltusLDAExample.scala](blob/master/src/main/scala/com/datascience/altus/AltusLDAExample.scala) is set up
primarily as code that can be run and manipulated by itself in the Workbench, while still compiling
as part of a runnable Spark application.

As such it requires running most, but not all, code in the file, as some code here is required for the
app packaging but will not work with the Workbench. Select and execute code within the two START/END blocks only.

The code builds and evaluates a few LDA models based on a subset of the data, and prints information
about the best model that was found.

## Scaling Up with Altus

With a working model selection process, it's possible to temporarily leverage much more computing power in
the cloud by spinning up a much larger transient cluster with Altus on AWS to try more models on the full data
set.

### Build the Spark App

The code is already set up to be packaged as a stand-alone JAR file that can be run with `spark-submit` or
Altus. It must first be built. This can be done in the Workbench's Terminal, even.

If [Apache Maven](https://maven.apache.org/) is not already available (and it typically is not), 
install it with a package manager. To install it manually, where this is not possible:

```bash
curl https://www.mirrorservice.org/sites/ftp.apache.org/maven/maven-3/3.5.0/binaries/apache-maven-3.5.0-bin.tar.gz | tar xz
alias mvn=./apache-maven-3.5.0/bin/mvn
```

To package the application JAR file:

```bash
mvn -Pcloudera -Pspark-deploy package
```

This produces a runnable JAR file like `altus-lda-example-1.0.0-jar-with-dependencies.jar` in `target/`.

### Get the Data

The data and JAR file both need to first be added to Amazon's S3 storage service. The same data files need
to exist in an S3 bucket that you have access to. Given the directory of files that was unpacked above to
upload to HDFS, it can be uploaded to an S3 bucket with:

```bash
aws s3 cp --recursive gutenberg s3://[your-bucket-here]/
```

The S3 bucket `s3://altus-cdsw-lda-example/` already contains this data, and can be used directly if desired
instead of uploading.

### Deploy to Altus

TODO (this spark-submit won't be how Altus does it)

```bash
spark2-submit \
  --master yarn --deploy-mode client \
  --driver-memory 4g \
  --num-executors 5 \
  --executor-cores 12 \
  --executor-memory 12g \
  --conf spark.yarn.executor.memoryOverhead=4g \
  --conf spark.dynamicAllocation.enabled=false \
  --conf spark.kryoserializer.buffer.max=256m \
  --class altus.AltusLDAExample \
  target/altus-lda-example-1.0.0-jar-with-dependencies.jar \
 	--dataDir hdfs:///user/ds/gutenberg \
 	--sampleRate 1.0 \
 	--kValues 5,10,25,100,250 \
 	--maxIter 100
```
