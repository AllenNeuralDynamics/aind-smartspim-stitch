#!/bin/bash

# BigStitcher Spark Runner - Optimized Version
# Created by Camilo Laiton, Jul 2025.
# This script runs BigStitcher Spark stitching with parameterized configuration

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# CONFIGURATION PARAMETERS FOR BIGSTITCHER SPARK
# =============================================================================

HOME=/root

# Java/JVM Configuration
JAVA_HEAP_SIZE="${JAVA_HEAP_SIZE:-24g}"
SPARK_MASTER="${SPARK_MASTER:-local[20]}"
SPARK_THREADS="${SPARK_THREADS:-20}"

# Maven Repository Path
M2_REPO="${M2_REPO:-$HOME/.m2/repository}"

# Main JAR Configuration
BIGSTITCHER_VERSION="${BIGSTITCHER_VERSION:-0.1.0-SNAPSHOT}"
MAIN_JAR_PATH="${M2_REPO}/net/preibisch/BigStitcher-Spark/${BIGSTITCHER_VERSION}/BigStitcher-Spark-${BIGSTITCHER_VERSION}.jar"

# Main Class
MAIN_CLASS="${MAIN_CLASS:-net.preibisch.bigstitcher.spark.SparkPairwiseStitching}"

# Additional JVM Options
JVM_OPTS="${JVM_OPTS:-}"

# =============================================================================
# VERY LARGE DEPENDENCY DEFINITIONS SET UP IN THE ORIGINAL FILE DIVIDED
# =============================================================================

# Core BigDataViewer Dependencies
declare -a BDV_DEPS=(
    "sc/fiji/bigdataviewer-core/10.6.3/bigdataviewer-core-10.6.3.jar"
    "sc/fiji/bigdataviewer-vistools/1.0.0-beta-36/bigdataviewer-vistools-1.0.0-beta-36.jar"
    "org/bigdataviewer/bigdataviewer-n5/1.0.1/bigdataviewer-n5-1.0.1.jar"
)

# ImgLib2 Dependencies
declare -a IMGLIB2_DEPS=(
    "net/imglib2/imglib2/7.1.4/imglib2-7.1.4.jar"
    "net/imglib2/imglib2-algorithm/0.17.2/imglib2-algorithm-0.17.2.jar"
    "net/imglib2/imglib2-algorithm-fft/0.2.1/imglib2-algorithm-fft-0.2.1.jar"
    "net/imglib2/imglib2-algorithm-gpl/0.3.1/imglib2-algorithm-gpl-0.3.1.jar"
    "net/imglib2/imglib2-cache/1.0.0-beta-19/imglib2-cache-1.0.0-beta-19.jar"
    "net/imglib2/imglib2-ij/2.0.3/imglib2-ij-2.0.3.jar"
    "net/imglib2/imglib2-label-multisets/0.15.1/imglib2-label-multisets-0.15.1.jar"
    "net/imglib2/imglib2-realtransform/4.0.3/imglib2-realtransform-4.0.3.jar"
    "net/imglib2/imglib2-roi/0.15.1/imglib2-roi-0.15.1.jar"
)

# N5 Storage Dependencies
declare -a N5_DEPS=(
    "org/janelia/saalfeldlab/n5/3.3.0/n5-3.3.0.jar"
    "org/janelia/saalfeldlab/n5-universe/1.6.0/n5-universe-1.6.0.jar"
    "org/janelia/saalfeldlab/n5-aws-s3/4.2.1/n5-aws-s3-4.2.1.jar"
    "org/janelia/saalfeldlab/n5-blosc/1.1.1/n5-blosc-1.1.1.jar"
    "org/janelia/saalfeldlab/n5-google-cloud/4.1.1/n5-google-cloud-4.1.1.jar"
    "org/janelia/saalfeldlab/n5-hdf5/2.2.0/n5-hdf5-2.2.0.jar"
    "org/janelia/saalfeldlab/n5-imglib2/7.0.2/n5-imglib2-7.0.2.jar"
    "org/janelia/saalfeldlab/n5-zarr/1.3.5/n5-zarr-1.3.5.jar"
    "org/janelia/n5-zstandard/1.0.2/n5-zstandard-1.0.2.jar"
)

# Apache Spark Dependencies
declare -a SPARK_DEPS=(
    "org/apache/spark/spark-core_2.12/3.3.2/spark-core_2.12-3.3.2.jar"
    "org/apache/spark/spark-launcher_2.12/3.3.2/spark-launcher_2.12-3.3.2.jar"
    "org/apache/spark/spark-kvstore_2.12/3.3.2/spark-kvstore_2.12-3.3.2.jar"
    "org/apache/spark/spark-network-common_2.12/3.3.2/spark-network-common_2.12-3.3.2.jar"
    "org/apache/spark/spark-network-shuffle_2.12/3.3.2/spark-network-shuffle_2.12-3.3.2.jar"
    "org/apache/spark/spark-unsafe_2.12/3.3.2/spark-unsafe_2.12-3.3.2.jar"
    "org/apache/spark/spark-tags_2.12/3.3.2/spark-tags_2.12-3.3.2.jar"
    "org/spark-project/spark/unused/1.0.0/unused-1.0.0.jar"
)

# Scala Dependencies
declare -a SCALA_DEPS=(
    "org/scala-lang/scala-library/2.12.15/scala-library-2.12.15.jar"
    "org/scala-lang/scala-reflect/2.12.15/scala-reflect-2.12.15.jar"
    "org/scala-lang/modules/scala-xml_2.12/1.2.0/scala-xml_2.12-1.2.0.jar"
)

# Jackson JSON Dependencies
declare -a JACKSON_DEPS=(
    "com/fasterxml/jackson/core/jackson-core/2.18.0/jackson-core-2.18.0.jar"
    "com/fasterxml/jackson/core/jackson-databind/2.13.4/jackson-databind-2.13.4.jar"
    "com/fasterxml/jackson/core/jackson-annotations/2.18.0/jackson-annotations-2.18.0.jar"
    "com/fasterxml/jackson/dataformat/jackson-dataformat-cbor/2.18.0/jackson-dataformat-cbor-2.18.0.jar"
    "com/fasterxml/jackson/module/jackson-module-scala_2.12/2.13.4/jackson-module-scala_2.12-2.13.4.jar"
)

# Google Cloud Dependencies
declare -a GCLOUD_DEPS=(
    "com/google/cloud/google-cloud-storage/2.34.0/google-cloud-storage-2.34.0.jar"
    "com/google/cloud/google-cloud-core/2.33.0/google-cloud-core-2.33.0.jar"
    "com/google/cloud/google-cloud-core-http/2.33.0/google-cloud-core-http-2.33.0.jar"
    "com/google/cloud/google-cloud-core-grpc/2.33.0/google-cloud-core-grpc-2.33.0.jar"
    "com/google/cloud/google-cloud-resourcemanager/1.38.0/google-cloud-resourcemanager-1.38.0.jar"
)

# AWS Dependencies
declare -a AWS_DEPS=(
    "com/amazonaws/aws-java-sdk-s3/1.12.772/aws-java-sdk-s3-1.12.772.jar"
    "com/amazonaws/aws-java-sdk-kms/1.12.772/aws-java-sdk-kms-1.12.772.jar"
    "com/amazonaws/aws-java-sdk-core/1.12.772/aws-java-sdk-core-1.12.772.jar"
    "com/amazonaws/jmespath-java/1.12.772/jmespath-java-1.12.772.jar"
)

# Apache Commons Dependencies
declare -a COMMONS_DEPS=(
    "org/apache/commons/commons-lang3/3.17.0/commons-lang3-3.17.0.jar"
    "org/apache/commons/commons-math3/3.6.1/commons-math3-3.6.1.jar"
    "org/apache/commons/commons-text/1.12.0/commons-text-1.12.0.jar"
    "org/apache/commons/commons-compress/1.27.1/commons-compress-1.27.1.jar"
    "org/apache/commons/commons-collections4/4.5.0-M2/commons-collections4-4.5.0-M2.jar"
    "org/apache/commons/commons-crypto/1.1.0/commons-crypto-1.1.0.jar"
    "commons-io/commons-io/2.17.0/commons-io-2.17.0.jar"
    "commons-codec/commons-codec/1.17.1/commons-codec-1.17.1.jar"
    "commons-collections/commons-collections/3.2.2/commons-collections-3.2.2.jar"
    "commons-lang/commons-lang/2.6/commons-lang-2.6.jar"
    "commons-logging/commons-logging/1.3.4/commons-logging-1.3.4.jar"
)

# Compression Libraries
declare -a COMPRESSION_DEPS=(
    "org/tukaani/xz/1.10/xz-1.10.jar"
    "org/lz4/lz4-java/1.9-inv/lz4-java-1.9-inv.jar"
    "org/xerial/snappy/snappy-java/1.1.8.4/snappy-java-1.1.8.4.jar"
    "com/github/luben/zstd-jni/1.5.6-6/zstd-jni-1.5.6-6.jar"
    "com/ning/compress-lzf/1.1/compress-lzf-1.1.jar"
    "org/lasersonlab/jblosc/1.0.1/jblosc-1.0.1.jar"
)

# Logging Dependencies
declare -a LOGGING_DEPS=(
    "org/slf4j/slf4j-api/1.7.36/slf4j-api-1.7.36.jar"
    "org/apache/logging/log4j/log4j-api/2.20.0/log4j-api-2.20.0.jar"
    "org/apache/logging/log4j/log4j-core/2.20.0/log4j-core-2.20.0.jar"
    "org/apache/logging/log4j/log4j-slf4j-impl/2.17.2/log4j-slf4j-impl-2.17.2.jar"
    "org/apache/logging/log4j/log4j-1.2-api/2.17.2/log4j-1.2-api-2.17.2.jar"
    "ch/qos/logback/logback-core/1.2.12/logback-core-1.2.12.jar"
)

# GRPC Dependencies
declare -a GRPC_DEPS=(
    "io/grpc/grpc-context/1.68.0/grpc-context-1.68.0.jar"
    "io/grpc/grpc-api/1.68.0/grpc-api-1.68.0.jar"
    "io/grpc/grpc-core/1.68.0/grpc-core-1.68.0.jar"
    "io/grpc/grpc-stub/1.68.0/grpc-stub-1.68.0.jar"
    "io/grpc/grpc-netty-shaded/1.68.0/grpc-netty-shaded-1.68.0.jar"
    "io/grpc/grpc-protobuf/1.68.0/grpc-protobuf-1.68.0.jar"
    "io/grpc/grpc-protobuf-lite/1.61.1/grpc-protobuf-lite-1.61.1.jar"
    "io/grpc/grpc-inprocess/1.68.0/grpc-inprocess-1.68.0.jar"
    "io/grpc/grpc-alts/1.68.0/grpc-alts-1.68.0.jar"
    "io/grpc/grpc-grpclb/1.68.0/grpc-grpclb-1.68.0.jar"
    "io/grpc/grpc-auth/1.68.0/grpc-auth-1.68.0.jar"
    "io/grpc/grpc-googleapis/1.68.0/grpc-googleapis-1.68.0.jar"
    "io/grpc/grpc-xds/1.68.0/grpc-xds-1.68.0.jar"
    "io/grpc/grpc-util/1.61.1/grpc-util-1.61.1.jar"
    "io/grpc/grpc-services/1.68.0/grpc-services-1.68.0.jar"
    "io/grpc/grpc-rls/1.61.1/grpc-rls-1.61.1.jar"
)

# ImageJ/Fiji Dependencies
declare -a IMAGEJ_DEPS=(
    "net/imagej/ij/1.54g/ij-1.54g.jar"
    "net/imagej/ij1-patcher/1.2.6/ij1-patcher-1.2.6.jar"
    "net/imagej/imagej-common/2.1.1/imagej-common-2.1.1.jar"
    "sc/fiji/Fiji_Plugins/3.1.3/Fiji_Plugins-3.1.3.jar"
    "sc/fiji/fiji-lib/2.1.3/fiji-lib-2.1.3.jar"
    "sc/fiji/spim_data/2.3.5/spim_data-2.3.5.jar"
)

# Math and Scientific Libraries
declare -a MATH_DEPS=(
    "gov/nist/math/jama/1.0.3/jama-1.0.3.jar"
    "org/ojalgo/ojalgo/45.1.1/ojalgo-45.1.1.jar"
    "mpicbg/mpicbg/1.6.0/mpicbg-1.6.0.jar"
    "edu/mines/mines-jtk/20151125/mines-jtk-20151125.jar"
    "jitk/jitk-tps/3.0.4/jitk-tps-3.0.4.jar"
)

# EJML (Linear Algebra) Dependencies
declare -a EJML_DEPS=(
    "org/ejml/ejml-all/0.41/ejml-all-0.41.jar"
    "org/ejml/ejml-core/0.41/ejml-core-0.41.jar"
    "org/ejml/ejml-fdense/0.41/ejml-fdense-0.41.jar"
    "org/ejml/ejml-ddense/0.41/ejml-ddense-0.41.jar"
    "org/ejml/ejml-cdense/0.41/ejml-cdense-0.41.jar"
    "org/ejml/ejml-zdense/0.41/ejml-zdense-0.41.jar"
    "org/ejml/ejml-dsparse/0.41/ejml-dsparse-0.41.jar"
    "org/ejml/ejml-simple/0.41/ejml-simple-0.41.jar"
    "org/ejml/ejml-fsparse/0.41/ejml-fsparse-0.41.jar"
)

# Preibisch Lab Dependencies
declare -a PREIBISCH_DEPS=(
    "net/preibisch/multiview-reconstruction/5.2.5/multiview-reconstruction-5.2.5.jar"
    "net/preibisch/multiview-simulation/0.2.0/multiview-simulation-0.2.0.jar"
    "net/preibisch/BigStitcher/2.3.4/BigStitcher-2.3.4.jar"
)

# Additional Utility Dependencies
declare -a UTIL_DEPS=(
    "org/scijava/scijava-common/2.99.0/scijava-common-2.99.0.jar"
    "org/scijava/parsington/3.1.0/parsington-3.1.0.jar"
    "org/scijava/scijava-listeners/1.0.0-beta-3/scijava-listeners-1.0.0-beta-3.jar"
    "org/scijava/ui-behaviour/2.0.8/ui-behaviour-2.0.8.jar"
    "org/scijava/scijava-optional/1.0.1/scijava-optional-1.0.1.jar"
    "org/scijava/scijava-table/1.0.2/scijava-table-1.0.2.jar"
    "org/scijava/native-lib-loader/2.5.0/native-lib-loader-2.5.0.jar"
    "cisd/jhdf5/19.04.1/jhdf5-19.04.1.jar"
    "cisd/base/18.09.0/base-18.09.0.jar"
    "com/google/guava/guava/33.3.1-jre/guava-33.3.1-jre.jar"
    "com/google/guava/failureaccess/1.0.2/failureaccess-1.0.2.jar"
    "com/google/guava/listenablefuture/9999.0-empty-to-avoid-conflict-with-guava/listenablefuture-9999.0-empty-to-avoid-conflict-with-guava.jar"
    "com/google/j2objc/j2objc-annotations/3.0.0/j2objc-annotations-3.0.0.jar"
    "args4j/args4j/2.33/args4j-2.33.jar"
    "info/picocli/picocli/4.7.6/picocli-4.7.6.jar"
    "se/sawano/java/alphanumeric-comparator/1.4.1/alphanumeric-comparator-1.4.1.jar"
    "net/java/dev/jna/jna/5.7.0/jna-5.7.0.jar"
)

# Remaining Dependencies (miscellaneous)
declare -a MISC_DEPS=(
    "com/formdev/flatlaf/3.5.1/flatlaf-3.5.1.jar"
    "com/google/code/gson/gson/2.10.1/gson-2.10.1.jar"
    "com/miglayout/miglayout-swing/5.3/miglayout-swing-5.3.jar"
    "com/miglayout/miglayout-core/5.3/miglayout-core-5.3.jar"
    "dev/dirs/directories/26/directories-26.jar"
    "net/sf/trove4j/trove4j/3.0.3/trove4j-3.0.3.jar"
    "org/jdom/jdom2/2.0.6.1/jdom2-2.0.6.1.jar"
    "org/yaml/snakeyaml/2.3/snakeyaml-2.3.jar"
    "com/github/ben-manes/caffeine/caffeine/2.9.3/caffeine-2.9.3.jar"
    "org/checkerframework/checker-qual/3.48.0/checker-qual-3.48.0.jar"
    "com/google/errorprone/error_prone_annotations/2.32.0/error_prone_annotations-2.32.0.jar"
)

# =============================================================================
# FUNCTIONS TO BUILD CLASSPATHS AND VALIDATION OF CONFIGS
# =============================================================================

# Function to build classpath from dependency arrays
build_classpath() {
    local classpath="$MAIN_JAR_PATH"
    
    # Combine all dependency arrays
    local all_deps=(
        "${BDV_DEPS[@]}"
        "${IMGLIB2_DEPS[@]}"
        "${N5_DEPS[@]}"
        "${SPARK_DEPS[@]}"
        "${SCALA_DEPS[@]}"
        "${JACKSON_DEPS[@]}"
        "${GCLOUD_DEPS[@]}"
        "${AWS_DEPS[@]}"
        "${COMMONS_DEPS[@]}"
        "${COMPRESSION_DEPS[@]}"
        "${LOGGING_DEPS[@]}"
        "${GRPC_DEPS[@]}"
        "${IMAGEJ_DEPS[@]}"
        "${MATH_DEPS[@]}"
        "${EJML_DEPS[@]}"
        "${PREIBISCH_DEPS[@]}"
        "${UTIL_DEPS[@]}"
        "${MISC_DEPS[@]}"
    )
    
    # Additional dependencies that need full paths
    local additional_deps=(
        "com/google/http-client/google-http-client-jackson2/1.45.0/google-http-client-jackson2-1.45.0.jar"
        "com/google/http-client/google-http-client-gson/1.45.0/google-http-client-gson-1.45.0.jar"
        "com/google/api-client/google-api-client/2.7.0/google-api-client-2.7.0.jar"
        "com/google/oauth-client/google-oauth-client/1.36.0/google-oauth-client-1.36.0.jar"
        "com/google/http-client/google-http-client-apache-v2/1.45.0/google-http-client-apache-v2-1.45.0.jar"
        "com/google/apis/google-api-services-storage/v1-rev20240209-2.0.0/google-api-services-storage-v1-rev20240209-2.0.0.jar"
        "com/google/auto/value/auto-value-annotations/1.11.0/auto-value-annotations-1.11.0.jar"
        "com/google/http-client/google-http-client-appengine/1.45.0/google-http-client-appengine-1.45.0.jar"
        "com/google/api/gax-httpjson/2.54.1/gax-httpjson-2.54.1.jar"
        "com/google/api/gax/2.54.1/gax-2.54.1.jar"
        "com/google/api/gax-grpc/2.54.1/gax-grpc-2.54.1.jar"
        "org/conscrypt/conscrypt-openjdk-uber/2.5.2/conscrypt-openjdk-uber-2.5.2.jar"
        "com/google/auth/google-auth-library-credentials/1.27.0/google-auth-library-credentials-1.27.0.jar"
        "com/google/auth/google-auth-library-oauth2-http/1.27.0/google-auth-library-oauth2-1.27.0.jar"
        "com/google/api/api-common/2.37.1/api-common-2.37.1.jar"
        "io/opencensus/opencensus-api/0.31.1/opencensus-api-0.31.1.jar"
        "io/opencensus/opencensus-contrib-http-util/0.31.1/opencensus-contrib-http-util-0.31.1.jar"
        "io/opencensus/opencensus-proto/0.2.0/opencensus-proto-0.2.0.jar"
        "com/google/api/grpc/proto-google-iam-v1/1.40.1/proto-google-iam-v1-1.40.1.jar"
        "com/google/protobuf/protobuf-java-util/4.28.2/protobuf-java-util-4.28.2.jar"
        "com/google/protobuf/protobuf-java/4.28.2/protobuf-java-4.28.2.jar"
        "com/google/api/grpc/proto-google-common-protos/2.45.1/proto-google-common-protos-2.45.1.jar"
        "org/threeten/threetenbp/1.7.0/threetenbp-1.7.0.jar"
        "com/google/api/grpc/proto-google-cloud-storage-v2/2.34.0-alpha/proto-google-cloud-storage-v2-2.34.0-alpha.jar"
        "com/google/api/grpc/grpc-google-cloud-storage-v2/2.34.0-alpha/grpc-google-cloud-storage-v2-2.34.0-alpha.jar"
        "com/google/api/grpc/gapic-google-cloud-storage-v2/2.34.0-alpha/gapic-google-cloud-storage-v2-2.34.0-alpha.jar"
        "com/google/android/annotations/4.1.1.4/annotations-4.1.1.4.jar"
        "org/codehaus/mojo/animal-sniffer-annotations/1.23/animal-sniffer-annotations-1.23.jar"
        "io/perfmark/perfmark-api/0.27.0/perfmark-api-0.27.0.jar"
        "com/google/re2j/re2j/1.7/re2j-1.7.jar"
        "com/google/api/grpc/proto-google-cloud-resourcemanager-v3/1.38.0/proto-google-cloud-resourcemanager-v3-1.38.0.jar"
        "com/google/apis/google-api-services-cloudresourcemanager/v1-rev20240128-2.0.0/google-api-services-cloudresourcemanager-v1-rev20240128-2.0.0.jar"
        "org/apache/httpcomponents/httpcore/4.4.16/httpcore-4.4.16.jar"
        "org/apache/httpcomponents/httpclient/4.5.14/httpclient-4.5.14.jar"
        "org/apache/httpcomponents/httpmime/4.5.14/httpmime-4.5.14.jar"
        "net/thisptr/jackson-jq/1.0.0-preview.20191208/jackson-jq-1.0.0-preview.20191208.jar"
        "org/jruby/joni/joni/2.2.1/joni-2.2.1.jar"
        "org/jruby/jcodings/jcodings/1.0.58/jcodings-1.0.58.jar"
        "org/apache/avro/avro/1.11.0/avro-1.11.0.jar"
        "org/apache/avro/avro-mapred/1.11.0/avro-mapred-1.11.0.jar"
        "org/apache/avro/avro-ipc/1.11.0/avro-ipc-1.11.0.jar"
        "com/twitter/chill_2.12/0.10.0/chill_2.12-0.10.0.jar"
        "com/esotericsoftware/kryo-shaded/4.0.2/kryo-shaded-4.0.2.jar"
        "com/esotericsoftware/minlog/1.3.1/minlog-1.3.1.jar"
        "org/objenesis/objenesis/3.4/objenesis-3.4.jar"
        "com/twitter/chill-java/0.10.0/chill-java-0.10.0.jar"
        "org/apache/xbean/xbean-asm9-shaded/4.20/xbean-asm9-shaded-4.20.jar"
        "org/apache/hadoop/hadoop-client-api/3.3.2/hadoop-client-api-3.3.2.jar"
        "org/apache/hadoop/hadoop-client-runtime/3.3.2/hadoop-client-runtime-3.3.2.jar"
        "org/fusesource/leveldbjni/leveldbjni-all/1.8/leveldbjni-all-1.8.jar"
        "org/rocksdb/rocksdbjni/6.20.3/rocksdbjni-6.20.3.jar"
        "com/google/crypto/tink/tink/1.6.1/tink-1.6.1.jar"
        "javax/activation/activation/1.1.1/activation-1.1.1.jar"
        "org/apache/curator/curator-recipes/2.13.0/curator-recipes-2.13.0.jar"
        "org/apache/curator/curator-framework/2.13.0/curator-framework-2.13.0.jar"
        "org/apache/curator/curator-client/2.13.0/curator-client-2.13.0.jar"
        "org/apache/zookeeper/zookeeper/3.6.2/zookeeper-3.6.2.jar"
        "org/apache/zookeeper/zookeeper-jute/3.6.2/zookeeper-jute-3.6.2.jar"
        "org/apache/yetus/audience-annotations/0.5.0/audience-annotations-0.5.0.jar"
        "jakarta/servlet/jakarta.servlet-api/4.0.3/jakarta.servlet-api-4.0.3.jar"
        "org/roaringbitmap/RoaringBitmap/0.9.25/RoaringBitmap-0.9.25.jar"
        "org/roaringbitmap/shims/0.9.25/shims-0.9.25.jar"
        "org/json4s/json4s-jackson_2.12/3.7.0-M11/json4s-jackson_2.12-3.7.0-M11.jar"
        "org/json4s/json4s-core_2.12/3.7.0-M11/json4s-core_2.12-3.7.0-M11.jar"
        "org/json4s/json4s-ast_2.12/3.7.0-M11/json4s-ast_2.12-3.7.0-M11.jar"
        "org/json4s/json4s-scalap_2.12/3.7.0-M11/json4s-scalap_2.12-3.7.0-M11.jar"
        "org/glassfish/jersey/core/jersey-common/2.36/jersey-common-2.36.jar"
        "jakarta/ws/rs/jakarta.ws.rs-api/2.1.6/jakarta.ws.rs-api-2.1.6.jar"
        "jakarta/annotation/jakarta.annotation-api/1.3.5/jakarta.annotation-api-1.3.5.jar"
        "org/glassfish/hk2/external/jakarta.inject/2.6.1/jakarta.inject-2.6.1.jar"
        "org/glassfish/hk2/osgi-resource-locator/1.0.3/osgi-resource-locator-1.0.3.jar"
        "org/glassfish/jersey/core/jersey-server/2.36/jersey-server-2.36.jar"
        "jakarta/validation/jakarta.validation-api/2.0.2/jakarta.validation-api-2.0.2.jar"
        "org/glassfish/jersey/containers/jersey-container-servlet/2.36/jersey-container-servlet-2.36.jar"
        "org/glassfish/jersey/containers/jersey-container-servlet-core/2.36/jersey-container-servlet-core-2.36.jar"
        "org/glassfish/jersey/inject/jersey-hk2/2.36/jersey-hk2-2.36.jar"
        "org/glassfish/hk2/hk2-locator/2.6.1/hk2-locator-2.6.1.jar"
        "org/glassfish/hk2/hk2-api/2.6.1/hk2-api-2.6.1.jar"
        "org/glassfish/hk2/hk2-utils/2.6.1/hk2-utils-2.6.1.jar"
        "org/javassist/javassist/3.30.2-GA/javassist-3.30.2-GA.jar"
        "com/clearspring/analytics/stream/2.9.6/stream-2.9.6.jar"
        "io/dropwizard/metrics/metrics-core/4.2.7/metrics-core-4.2.7.jar"
        "io/dropwizard/metrics/metrics-jvm/4.2.7/metrics-jvm-4.2.7.jar"
        "io/dropwizard/metrics/metrics-json/4.2.7/metrics-json-4.2.7.jar"
        "io/dropwizard/metrics/metrics-graphite/4.2.7/metrics-graphite-4.2.7.jar"
        "io/dropwizard/metrics/metrics-jmx/4.2.7/metrics-jmx-4.2.7.jar"
        "com/thoughtworks/paranamer/paranamer/2.8/paranamer-2.8.jar"
        "org/apache/ivy/ivy/2.5.2/ivy-2.5.2.jar"
        "oro/oro/2.0.8/oro-2.0.8.jar"
        "net/razorvine/pickle/1.2/pickle-1.2.jar"
        "net/sf/py4j/py4j/0.10.9.5/py4j-0.10.9.5.jar"
        "edu/ucar/udunits/4.3.18/udunits-4.3.18.jar"
        "org/jfree/jfreechart/1.5.5/jfreechart-1.5.5.jar"
        "gov/nist/isg/pyramidio/1.1.0/pyramidio-1.1.0.jar"
        "gov/nist/isg/generic-archiver/1.1.0/generic-archiver-1.1.0.jar"
        "com/github/jai-imageio/jai-imageio-core/1.3.1/jai-imageio-core-1.3.1.jar"
        "io/netty/netty-all/4.1.68.Final/netty-all-4.1.68.Final.jar"
        "io/netty/netty/3.9.9.Final/netty-3.9.9.Final.jar"
        "org/openmicroscopy/ome-common/6.0.4/ome-common-6.0.4.jar"
        "io/minio/minio/5.0.2/minio-5.0.2.jar"
        "com/google/http-client/google-http-client-xml/1.45.0/google-http-client-xml-1.45.0.jar"
        "com/google/http-client/google-http-client/1.45.0/google-http-client-1.45.0.jar"
        "xpp3/xpp3/1.1.4c/xpp3-1.1.4c.jar"
        "com/squareup/okhttp3/okhttp/4.12.0/okhttp-4.12.0.jar"
        "org/jetbrains/kotlin/kotlin-stdlib-jdk8/1.9.22/kotlin-stdlib-jdk8-1.9.22.jar"
        "org/jetbrains/kotlin/kotlin-stdlib/1.9.22/kotlin-stdlib-1.9.22.jar"
        "org/jetbrains/annotations/13.0/annotations-13.0.jar"
        "org/jetbrains/kotlin/kotlin-stdlib-jdk7/1.9.22/kotlin-stdlib-jdk7-1.9.22.jar"
        "com/squareup/okio/okio/3.9.1/okio-3.9.1.jar"
        "com/squareup/okio/okio-jvm/3.9.1/okio-jvm-3.9.1.jar"
        "joda-time/joda-time/2.13.0/joda-time-2.13.0.jar"
        "ome/formats-bsd/6.5.1/formats-bsd-6.5.1.jar"
        "org/openmicroscopy/ome-xml/6.3.6/ome-xml-6.3.6.jar"
        "org/openmicroscopy/specification/6.3.6/specification-6.3.6.jar"
        "ome/formats-api/7.3.1/formats-api-7.3.1.jar"
        "org/openmicroscopy/ome-codecs/0.3.0/ome-codecs-0.3.0.jar"
        "org/openmicroscopy/ome-jai/0.1.0/ome-jai-0.1.0.jar"
        "ome/turbojpeg/6.5.1/turbojpeg-6.5.1.jar"
        "com/jgoodies/jgoodies-forms/1.7.2/jgoodies-forms-1.7.2.jar"
        "com/jgoodies/jgoodies-common/1.7.0/jgoodies-common-1.7.0.jar"
        "org/perf4j/perf4j/0.9.16/perf4j-0.9.16.jar"
        "com/drewnoakes/metadata-extractor/2.11.0/metadata-extractor-2.11.0.jar"
        "com/adobe/xmp/xmpcore/5.1.3/xmpcore-5.1.3.jar"
        "ome/jxrlib-all/0.2.4/jxrlib-all-0.2.4.jar"
        "xerces/xercesImpl/2.8.1/xercesImpl-2.8.1.jar"
        "xml-apis/xml-apis/1.3.03/xml-apis-1.3.03.jar"
        "ome/formats-gpl/6.5.1/formats-gpl-6.5.1.jar"
        "org/openmicroscopy/ome-mdbtools/5.3.2/ome-mdbtools-5.3.2.jar"
        "org/openmicroscopy/metakit/5.3.7/metakit-5.3.7.jar"
        "org/openmicroscopy/ome-poi/5.3.9/ome-poi-5.3.9.jar"
        "edu/ucar/cdm/4.6.13/cdm-4.6.13.jar"
        "edu/ucar/httpservices/4.6.13/httpservices-4.6.13.jar"
        "com/mchange/c3p0/0.9.5.3/c3p0-0.9.5.3.jar"
        "com/mchange/mchange-commons-java/0.2.15/mchange-commons-java-0.2.15.jar"
        "woolz/JWlz/1.4.0/JWlz-1.4.0.jar"
        "org/json/json/20240303/json-20240303.jar"
        "junit/junit/4.13.2/junit-4.13.2.jar"
        "org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar"
        "com/google/code/findbugs/jsr305/3.0.2/jsr305-3.0.2.jar"
    )
    
    # Build classpath
    for dep in "${all_deps[@]}" "${additional_deps[@]}"; do
        classpath="$classpath:$M2_REPO/$dep"
    done
    
    echo "$classpath"
}

# Function to check if main JAR exists
check_main_jar() {
    if [[ ! -f "$MAIN_JAR_PATH" ]]; then
        echo "ERROR: Main JAR not found at: $MAIN_JAR_PATH" >&2
        echo "Please check BIGSTITCHER_VERSION and M2_REPO variables." >&2
        exit 1
    fi
}

# Function to validate Spark configuration
validate_spark_config() {
    if [[ "$SPARK_MASTER" == "local["* ]]; then
        # Extract thread count from local[N] format
        local threads=$(echo "$SPARK_MASTER" | sed 's/local\[\([0-9]*\)\]/\1/')
        if [[ "$threads" != "$SPARK_THREADS" ]]; then
            echo "WARNING: SPARK_MASTER threads ($threads) != SPARK_THREADS ($SPARK_THREADS)" >&2
        fi
    fi
}

# Function to print configuration
print_config() {
    echo "=== BigStitcher Spark Configuration ==="
    echo "Java Heap Size: $JAVA_HEAP_SIZE"
    echo "Spark Master: $SPARK_MASTER"
    echo "Spark Threads: $SPARK_THREADS"
    echo "Maven Repository: $M2_REPO"
    echo "Main JAR: $MAIN_JAR_PATH"
    echo "Main Class: $MAIN_CLASS"
    echo "Additional JVM Options: ${JVM_OPTS:-None}"
    echo "========================================="
}

# Function to show usage
show_usage() {
    cat << EOF
BigStitcher Spark Runner - Optimized Version

USAGE:
    $0 [script-options] [-- | application-arguments...]

ENVIRONMENT VARIABLES:
    JAVA_HEAP_SIZE      Java heap size (default: 24g)
    SPARK_MASTER        Spark master URL (default: local[20])
    SPARK_THREADS       Number of Spark threads (default: 20)
    M2_REPO            Maven repository path (default: \$HOME/.m2/repository)
    BIGSTITCHER_VERSION BigStitcher version (default: 0.1.0-SNAPSHOT)
    MAIN_CLASS         Main class to run (default: net.preibisch.bigstitcher.spark.SparkPairwiseStitching)
    JVM_OPTS           Additional JVM options

SCRIPT OPTIONS:
    -h, --help         Show this help message
    -v, --verbose      Show configuration before running
    -c, --check        Check dependencies and exit
    --dry-run          Show command that would be executed without running it
    --                 Explicitly separate script options from application arguments

EXAMPLES:
    # Basic usage (first non-option argument switches to application args)
    $0 --xml input.xml -ds 4,4,4

    # With script options first
    $0 --verbose --dry-run --xml input.xml -ds 4,4,4

    # Using explicit separator
    $0 --verbose --dry-run -- --xml input.xml -ds 4,4,4

    # With custom configuration
    JAVA_HEAP_SIZE=32g SPARK_THREADS=32 $0 --xml input.xml -ds 4,4,4

    # Check dependencies
    $0 --check
EOF
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

# Parse command line arguments
VERBOSE=false
CHECK_ONLY=false
DRY_RUN=false

# Separate script options from application arguments
SCRIPT_ARGS=()
APP_ARGS=()
PARSING_SCRIPT_ARGS=true

while [[ $# -gt 0 ]]; do
    if [[ "$PARSING_SCRIPT_ARGS" == true ]]; then
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -c|--check)
                CHECK_ONLY=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --)
                # Explicit separator - everything after this goes to application
                shift
                PARSING_SCRIPT_ARGS=false
                ;;
            -*)
                # If it's a script option we don't recognize, assume it's for the application
                PARSING_SCRIPT_ARGS=false
                APP_ARGS+=("$1")
                shift
                ;;
            *)
                # First non-option argument - switch to application args
                PARSING_SCRIPT_ARGS=false
                APP_ARGS+=("$1")
                shift
                ;;
        esac
    else
        # All remaining arguments go to the application
        APP_ARGS+=("$1")
        shift
    fi
done

# Show configuration if verbose mode
if [[ "$VERBOSE" == true ]]; then
    print_config
fi

# Check main JAR exists
check_main_jar

# Validate Spark configuration
validate_spark_config

# Build classpath
echo "Building classpath..."
CLASSPATH=$(build_classpath)

if [[ "$CHECK_ONLY" == true ]]; then
    echo "Dependency check completed successfully."
    echo "Main JAR: $MAIN_JAR_PATH ✓"
    echo "Classpath built with $(echo "$CLASSPATH" | tr ':' '\n' | wc -l) entries ✓"
    exit 0
fi

# Build Java command
JAVA_CMD=(
    java
    -Xmx"$JAVA_HEAP_SIZE"
    -Dspark.master="$SPARK_MASTER"
)

# Add additional JVM options if specified
if [[ -n "$JVM_OPTS" ]]; then
    # Split JVM_OPTS on spaces and add to array
    read -ra JVM_OPTS_ARRAY <<< "$JVM_OPTS"
    JAVA_CMD+=("${JVM_OPTS_ARRAY[@]}")
fi

# Add classpath and main class
JAVA_CMD+=(
    -cp "$CLASSPATH"
    "$MAIN_CLASS"
)

# Add application arguments
JAVA_CMD+=("${APP_ARGS[@]}")

# Execute or show command
if [[ "$DRY_RUN" == true ]]; then
    echo "Command that would be executed:"
    printf '%s ' "${JAVA_CMD[@]}"
    echo
else
    echo "Starting BigStitcher Spark..."
    exec "${JAVA_CMD[@]}"
fi