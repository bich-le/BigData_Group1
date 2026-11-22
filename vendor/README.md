# Windows local Spark support

`hadoop-bare-naked-local-fs-0.1.0.jar` is the Apache-2.0 GlobalMentor Hadoop
Bare Naked Local FileSystem library. `mllib_pipeline.configure_local_filesystem`
loads it only on Windows when the JAR exists. It lets PySpark persist
the MLflow Spark model on Windows without `winutils.exe`.

Maven coordinate: `com.globalmentor:hadoop-bare-naked-local-fs:0.1.0`.

Verified SHA-1: `CD03DC0F6E2B8D8957D97D421E95D9CEAA16B06B`.
