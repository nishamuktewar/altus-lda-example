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
from the project mirrors, it's very large. A compressed archive of only the text files, with most 
duplicates and obsolete text files removed, in Parquet format, can be downloaded, decompressed, 
and uploaded to a directory like `/user/ds/gutenberg` on HDFS as follows:

```bash
curl https://storage.googleapis.com/altus-cdsw-lda-example/gutenberg.tar | tar xv   
hdfs dfs -put gutenberg /user/ds/
```

The data consists of `(path, text)` pairs, where "path" is the path from the original Gutenberg archive
and "text" is the text of the corresponding file.

If you really want to download all of the files directly, then just the text files can be `rsync`ed with:

```bash
rsync --include "*/" --include "*.txt" --exclude "*" -zarv aleph.gutenberg.org::gutenberg gutenberg/
```

... and then further pared down from there. But this isn't recommended.

### Checkout and Run Code in the Workbench

Start by creating a new project in the Workbench project based on the Git repo at
https://github.com/nisha/altus-lda-example and open it in the Workbench.

The file [spark-defaults.conf](blob/master/spark-defaults.conf) contains default resource sizing
for a moderately powerful 5-node cluster. This sizing can be increased or reduced to match your
cluster. If in doubt, delete this file in your project in order to use cluster defaults.

Run a Scala session with 4 CPUs and 8GB of RAM. (More, or less, depending on your driver resource size.)

[AltusLDAExample.scala](blob/master/src/main/scala/com/datascience/altus/AltusLDAExample.scala) is set up
primarily as code that can be run and manipulated by itself in the Workbench, while still compiling
as part of a runnable Spark application. As such it requires running most, but not all, code in the file, as some code here is required for the
app packaging but will not work with the Workbench. Select and execute code within the two START/END blocks only.

The code builds and evaluates a few LDA models based on a subset of the data, and prints information
about the best model that was found. It also saves the model to the path defined by `--outputPath` if set.

## Scaling Up with Altus

With a working model selection process, it's possible to temporarily leverage much more computing power in
the cloud by spinning up a much larger transient cluster with Altus on AWS to try more models on the full data
set.

### Build the Spark App

*Note:* You can skip this step and use the pre-built JAR file at 
`s3a://altus-cdsw-lda-example/altus-lda-example-1.0.0-jar-with-dependencies.jar`.

The code is already set up to be packaged as a stand-alone JAR file that can be run with `spark-submit` or
Altus. It must first be built. This can be done in the Workbench's Terminal, even.

If [Apache Maven](https://maven.apache.org/) is not already available (and it typically is not), 
install it with a package manager. To install it manually, where this is not possible:

```bash
mirror=$(curl -s https://www.apache.org/dyn/closer.lua?preferred=true)
curl "${mirror}/maven/maven-3/3.5.2/binaries/apache-maven-3.5.2-bin.tar.gz" | tar xz
alias mvn=./apache-maven-3.5.2/bin/mvn
```

To package the application JAR file, just `mvn package`. 
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

### Set Up AWS

Using Altus currently means using AWS or Microsoft Azure. We'll use AWS in this example.

You will need an AWS account, one that has billing set up, because
the following operations will incur AWS charges. You'll also need to know the Access Key and Secret Key for 
your account. These are available in the [AWS Console](https://aws.amazon.com/), in your account menu at the top right, 
under  "My Security Credentials".

Altus will also require an SSH key. Navigate to the EC2 service in AWS, and choose a zone like `us-east-2` (US East Ohio)
from the region menu at the top right. Under "Network & Security" at the left, see "Key Pairs". Create a new key pair called
`AltusLDAExampleKey`, and make sure to keep and secure the `AltusLDAExampleKey.pem` private key file that is
created.

### Set Up Altus

Log in to [Altus](https://www.cloudera.com/products/altus.html).

To use Altus, first, you must set up an Environment, if none are already available to you. 
Choose "Environments" from the left and click "Quickstart". Click "Create" next to "Amazon Web Services". 

Set the "Altus Environment Name" to `AltusLDAExample` and choose the same region in which you created the SSH keys above,
such as `us-east-2`. Leave other options at the default. Supply your AWS Access Key and Secret Key when prompted.

Leave "Archive work loads" suggested, but, the default bucket name is based on the Environment name, and may not
be available. Use "Customize Resource Names" at the right to customize the logging bucket if needed.

### Deploy to Altus

Return to the main Altus pane and choose "Jobs" from the left, and choose "Submit Jobs". Fill out the details as
follows.

Note that `--dataDir` controls where the input files are read from. An additional option, `--outputDir` (not shown)
causes the final model to be written, as well. This can be another S3 bucket, but must be one you have already
set up and have write access to.

- Job Settings
  - Submission: Single job
  - Job Type: Spark
  - Job Name: `AltusLDAExample`
  - Main Class: `altus.AltusLDARunner`
  - Jars: `s3a://altus-cdsw-lda-example/altus-lda-example-1.0.0-jar-with-dependencies.jar` (or your uploaded JAR)
  - Application Arguments:
    - `--dataDir=s3a://altus-cdsw-lda-example/gutenberg`
    - `--sampleRate=1.0`
    - `--kValues=25,100,200`
    - `--maxIter=20`
  - Spark Arguments:
```
    --driver-cores 1 --driver-memory 4g 
    --num-executors 8 
    --executor-cores 14 --executor-memory 16g 
    --conf spark.locality.wait=1s
    --conf spark.yarn.executor.memoryOverhead=4g 
    --conf spark.dynamicAllocation.enabled=false 
    --conf fs.s3a.access.key="[AWS Access Key]" 
    --conf fs.s3.awsSecretAccessKey="[AWS Secret Key]"
```
- Cluster Settings
  - Cluster: Create New
    - Uncheck "Terminate cluster once jobs complete" if you wish to try submitting several jobs, but don't forget to shut it down
  - Cluster Name: `AltusLDAExample-cluster`
  - Service Type: Spark 2.x
  - CDH Version: CDH 5.13 (Spark 2.2)
  - Environment: `AltusLDAExample`
- Node Configuration:
  - Worker: 3 x c4.4xlarge
  - Compute Worker: 5 x c4.4xlarge
    - Purchasing Option: Spot, $0.80 / hour
- Credentials:
  - SSH Private Key: (`.pem` file from SSH key above)
  - Cloudera Manager: (any credentials you like)

You can find application logs on S3 at a path like
`altus-lda-example-log-archive-bucket/AltusLDAExample-cluster-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/hdfs/logs`.

*Note:* Above, AWS credentials are specified on the command line. This means they may be logged, and visible 
to users who can browse the job's log files. For alternative ways of specifying credentials, see
https://www.cloudera.com/documentation/enterprise/latest/topics/spark_s3.html .

### Bonus: Altus CLI

If you left the cluster running, or, already have a cluster running in Altus, you can even launch this
job from the command line, and from within the Workbench Terminal, too.

First `pip install altuscli` to get access to the latest Altus CLI tools. 

You will also need an _Altus_ Access Key (separate from AWS) in order to invoke Altus commands from the
command line. In the Altus console, navigate to "My Account" at the top right. Under "Access Keys", choose
"Generate Access Key". You will get an Access Key and Secret Key. Save the secret key! Or keep the window
open.

In Terminal, `altus configure` and enter your Altus Access Key and Secret Key.

At this point, it should be possible to 
[submit a new job](https://www.cloudera.com/documentation/altus/topics/altaws_dejob_jobs_cli.html#submit_spark_job_cli)
with:

```
altus dataeng submit-jobs \
 --cluster-name AltusLDAExample-cluster \
 --jobs '{ "name": "AltusLDAExample",
           "sparkJob": {
             "jars": [
               "s3a://altus-cdsw-lda-example/altus-lda-example-1.0.0-jar-with-dependencies.jar"
             ],
             "applicationArguments": "...",
             "sparkArguments": "...",
             "mainClass": "altus.AltusLDARunner"
        }}'
```

The `list-jobs` and `terminate-jobs` can be used to manage jobs from the CLI; try `altus dataeng help`. 
It's even possible to create clusters from the command line!