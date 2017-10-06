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
from the project mirrors, it's very large. A compressed archive of only the text files in Parquet format
can be downloaded, decompressed, and uploaded to a directory like `/user/ds/gutenberg` on HDFS as follows:

```bash
curl https://storage.googleapis.com/altus-cdsw-lda-example/gutenberg-20170929.tgz | tar xz
hdfs dfs -put gutenberg /user/ds/
```

The data consists of `(path, text)` pairs, where "path" is the path from the original Gutenberg archive
and "text" is the text of the corresponding file.

### Checkout and Run Code in the Workbench

Start by creating a new project in the Workbench project based on the Git repo at
https://github.com/nisha/altus-lda-example and open it in the Workbench.

The file [spark-defaults.conf](blob/master/spark-defaults.conf) contains default resource sizing
for a moderately powerful 5-node cluster. This sizing can be increased or reduced to match your
cluster.

Run a Scala session with 4 CPUs and 8GB of RAM. (More, or less, depending on your driver resource size.)

[AltusLDAExample.scala](blob/master/src/main/scala/com/datascience/altus/AltusLDAExample.scala) is set up
primarily as code that can be run and manipulated by itself in the Workbench, while still compiling
as part of a runnable Spark application. As such it requires running most, but not all, code in the file, as some code here is required for the
app packaging but will not work with the Workbench. Select and execute code within the two START/END blocks only.

The code builds and evaluates a few LDA models based on a subset of the data, and prints information
about the best model that was found.

## Scaling Up with Altus

With a working model selection process, it's possible to temporarily leverage much more computing power in
the cloud by spinning up a much larger transient cluster with Altus on AWS to try more models on the full data
set.

### Set Up AWS

Using Altus currently means using AWS or Microsoft Azure. We'll use AWS in this example.

You will need an AWS account, one that has billing set up, because
the following operations will incur AWS charges. You'll also need to know the Access Key and Secret Key for 
your account. These are available the AWS Console account menu, under "My Security Credentials".

Altus will also require an SSH key. Navigate to the EC2 service in AWS, and choose a zone like `us-east-2` (Ohio)
from the region menu. Under "Network & Security" at the left, see "Key Pairs". Create a new key pair called
`AltusLDAExampleKey`, and make sure to keep and secure the `AltusLDAExampleKey.pem` private key file that is
created.

You'll also need to install the AWS CLI interface. In the Workbench Terminal, `pip install awscli`.
Add your credentials with `aws configure`. Note that this means your AWS credentials are stored in the
Workbench project!

### Build the Spark App

*Note:* You can skip this step and use the pre-built JAR file at 
`s3a://altus-cdsw-lda-example/altus-lda-example-1.0.0-jar-with-dependencies.jar`.

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

Upload it to S3, like so:

```bash
aws s3 cp target/altus-lda-example-1.0.0-jar-with-dependencies.jar s3://[your-bucket-here]/
```

### Get the Data

*Note:* You can skip this step and use the data already copied to `s3a://altus-cdsw-lda-example/`.

The data and JAR file both need to first be added to Amazon's S3 storage service. The same data files need
to exist in an S3 bucket that you have access to. Given the directory of files that was unpacked above to
upload to HDFS, it can be uploaded to an S3 bucket with:

```bash
aws s3 cp --recursive gutenberg s3://[your-bucket-here]/gutenberg
```

### Deploy to Altus

Log in to [Altus](https://www.cloudera.com/products/altus.html).

To use Altus, first, you must set up an Environment. Choose "Environments" from the left and click "Quickstart".
Click "Create" next to AWS. 

Set the Altus Environment Name to `AltusLDAExample` and choose the same region in which you created the SSH keys above,
such as `us-east-2`. Leave other options at the default. Supply your AWS Access Key and Secret Key when prompted.

Return to the main Altus pane and choose "Jobs" from the left, and choose "Submit Jobs". Fill out the details as
follows:

- Job Settings
  - Submission: Single job
  - Job Type: Spark
  - Job Name: `AltusLDAExample`
  - Main Class: `altus.AltusLDARunner`
  - Jars: `s3a://altus-cdsw-lda-example/altus-lda-example-1.0.0-jar-with-dependencies.jar` (or your uploaded JAR)
  - Application Arguments:
```
--dataDir=s3a://altus-cdsw-lda-example/gutenberg --sampleRate=1.0 --kValues=5,10,25,100,250 --maxIter=30
```
  - Spark Arguments:
```
--driver-memory 4g --num-executors 3 \
--executor-cores 15 --executor-memory 32g \
--conf spark.yarn.executor.memoryOverhead=16g \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.kryoserializer.buffer.max=256m \
--conf fs.s3a.access.key="[AWS Access Key]" \
--conf fs.s3.awsSecretAccessKey="[AWS Secret Key]"
```
- Cluster Settings
  - Cluster: Create New
  - Cluster Name: `AltusLDAExample-cluster`
  - Service Type: Spark 2.x
  - CDH Version: CDH 5.12
  - Environment: `AltusLDAExample`
- Node Config: 3x m4.4xlarge Workers
