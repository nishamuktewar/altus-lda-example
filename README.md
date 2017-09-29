# Clustering Books with LDA, Cloudera Data Science Workbench and Cloudera Altus

This brief example demonstrates clustering text documents using 
[Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) (LDA). In particular
this example uses a large corpus of public-domain texts from [Project Gutenberg](https://www.gutenberg.org/).

This shows how a data scientist might use the 
[Cloudera Data Science Workbench](https://www.cloudera.com/products/data-science-and-engineering/data-science-workbench.html) 
to interactively explore data and build models on a small cluster, and then leverage a much larger transient cluster
provisioned with [Cloudera Altus](https://www.cloudera.com/products/altus.html) to select and build the best, 
final model with additional computing power.

## Exploring with the Workbench

### Get Data

...

### Checkout and Run Code in the Workbench

Start by creating a new project in the Workbench based on the Git repo at
http://github.mtv.cloudera.com/nisha/altus-lda-example .

...

## Setup

If [Apache Maven](https://maven.apache.org/) is not already available, install it with a package manager.
To install it manually, where this is not possible:

```bash
curl https://www.mirrorservice.org/sites/ftp.apache.org/maven/maven-3/3.5.0/binaries/apache-maven-3.5.0-bin.tar.gz | tar xz
alias mvn=./apache-maven-3.5.0/bin/mvn
```

...

## Build

```bash
mvn -Pcloudera -Pspark-deploy package
```

This produces a runnable JAR file like `altus-lda-example-1.0.0-jar-with-dependencies.jar` in `target/`.

## Deploy

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
  --class com.cloudera.datascience.LDAExample \
  altus-lda-example-1.0.0-jar-with-dependencies.jar \
 	--dataDir hdfs:///user/ds/gutenberg \
 	--sampleRate 1.0 \
 	--kValues 5,10,25,100,250 \
 	--maxIter 100
```
