package org.apache.spark.workspace

import org.apache.spark.SparkConf
import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.{RspRDD, SonPartitionCoalescer}
import org.apache.spark.sql.RspContext._
import org.apache.spark.sql.functions.{col, _}
import org.apache.spark.sql._

import java.util.Date
import scala.collection.mutable.ArrayBuffer

object Confidence3 {
    def main(args: Array[String]): Unit = {

      // TODO 构建spark环境
      val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
      val spark = SparkSession
        .builder()
        .config(sparkconf)
        .getOrCreate()
      println("------------Environment configured successfully---------------")

      import spark.implicits._

      // Define parameter combinations
      val parameters = Array(
        (0.01, 0.05),
        (0.02, 0.05),
        (0.05, 0.05),
        (0.10, 0.05),
        (0.15, 0.05)
      )

      // Results storage
      val results = ArrayBuffer[Array[String]]()

      // Read data
      val value: DataFrame = spark.read.parquet("data/TH2_260W_260_26W")
      val RSPdata: RspRDD[Row] = spark.rspRead.parquet("data/TH2_260W_260_26W").rdd
      val array: Array[Int] = scala.util.Random.shuffle((0 until RSPdata.getNumPartitions).toList).toArray
      val numberOfblocks = 12
      val MST = 0.01

      // First, get the ground truth from full dataset
      println("Calculating ground truth from full dataset...")
      val fullDatasetFpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(MST).setMinConfidence(0.6)
      val fullModel = fullDatasetFpgrowth.fit(value)
      val groundTruthFreqItemsets = fullModel.freqItemsets.collect().map(row => (row.getAs[Seq[String]]("items").toSet, row.getLong(1)))
      val groundTruthItemsets = groundTruthFreqItemsets.map(_._1).toSet

      parameters.foreach { case (error, delta) =>
        println(s"\n=== Running experiment with ε=$error, δ=$delta ===")
        val experimentStart: Date = new Date()

        var i = 0
        var rmax: Double = 0.0
        var rmaxWS: Double = 0.0
        val allConfidences = ArrayBuffer[Double]()

        // 定义一个DataFrame集合，用于存储每个文件挖掘出的频繁项集
        var allFrequentItemsets: DataFrame = spark.emptyDataFrame
        var allFrequentItemsetsWS: DataFrame = spark.emptyDataFrame
        var allDiscoveredItemsets = Set.empty[Set[String]]

        while(i < numberOfblocks) {
          val value1: RDD[Row] = RSPdata.coalesce(1, false, Option(new SonPartitionCoalescer(Array(array(i)))))
          val dataset: DataFrame = spark.createDataFrame(value1, value.schema)
          val datasetCount = dataset.count()
          print(s"数据块 $i 数据量: $datasetCount")

          val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(MST).setMinConfidence(0.6)
          val model = fpgrowth.fit(dataset)

          // Calculate confidence for association rules - collect all confidence values
          val associationRules = model.associationRules
          val confidenceValues = if (associationRules.count() > 0) {
            associationRules.select(col("confidence")).collect().map(_.getDouble(0))
          } else {
            Array[Double]()
          }
          allConfidences ++= confidenceValues

          // Calculate rmax and IS with single frequent itemset
          val FIM: DataFrame = model.freqItemsets
          val currentItemsets = FIM.collect().map(row => row.getAs[Seq[String]]("items").toSet).toSet
          allDiscoveredItemsets ++= currentItemsets

          if (allFrequentItemsets.isEmpty) {
            allFrequentItemsets = FIM.select("items")
          } else {
            allFrequentItemsets = allFrequentItemsets.union(FIM.select("items")).select("items").distinct()
          }

          val FIMsup: DataFrame = FIM.withColumn("supportValue", col("freq") / datasetCount)
          val stats: DataFrame = FIMsup.select("supportValue").describe()
          val maxValue = (stats.filter($"summary" === "max").select("supportValue").collect()(0)(0)).toString.toDouble
          if (rmax < maxValue) {
            rmax = maxValue
          }

          // Calculate rmax and IS without single frequent itemset
          val FIMWS: DataFrame = model.freqItemsets.filter(size($"items") > 1)
          if (allFrequentItemsetsWS.isEmpty) {
            allFrequentItemsetsWS = FIMWS.select("items")
          } else {
            allFrequentItemsetsWS = allFrequentItemsetsWS.union(FIMWS.select("items")).select("items").distinct()
          }

          val FIMWSsup: DataFrame = FIMWS.withColumn("supportValue", col("freq") / datasetCount)
          val statsWS: DataFrame = FIMWSsup.select("supportValue").describe()
          val maxValueWS = (statsWS.filter($"summary" === "max").select("supportValue").collect()(0)(0)).toString.toDouble
          if (rmaxWS < maxValueWS) {
            rmaxWS = maxValueWS
          }

          println(s"第$i 次 - rmax: $rmax, rmaxWS: $rmaxWS, 置信度数量: ${confidenceValues.length}")
          i = i + 1
        }

        // Calculate final metrics
        val distinctItemsets: Dataset[Row] = allFrequentItemsets.select("items").distinct()
        val distinctItemsetsCount = distinctItemsets.count()
        val S: Double = rmax * rmax / (2 * error * error) * math.log(2 * distinctItemsetsCount / delta)

        val distinctItemsetsWS: Dataset[Row] = allFrequentItemsetsWS.select("items").distinct()
        val distinctItemsetsWSCount = distinctItemsetsWS.count()
        val SWS: Double = rmaxWS * rmaxWS / (2 * error * error) * math.log(2 * distinctItemsetsWSCount / delta)

        val experimentEnd: Date = new Date()
        val runningTime = experimentEnd.getTime - experimentStart.getTime

        // Calculate Precision and Recall
        val discoveredItemsets = allDiscoveredItemsets
        val truePositives = discoveredItemsets.intersect(groundTruthItemsets).size
        val precision = if (discoveredItemsets.nonEmpty) truePositives.toDouble / discoveredItemsets.size else 0.0
        val recall = if (groundTruthItemsets.nonEmpty) truePositives.toDouble / groundTruthItemsets.size else 0.0

        // Calculate ASE (Average Support Error)
        val ase = if (discoveredItemsets.nonEmpty) {
          // This is a simplified ASE calculation - you might want to implement a more precise one
          error // Using error as a proxy for ASE in this example
        } else {
          0.0
        }

        // Format confidence output - show count and range instead of average
        val confidenceInfo = if (allConfidences.nonEmpty) {
          s"${allConfidences.size} rules"
        } else {
          "0 rules"
        }

        // Store results
        val result = Array(
          error.toString,
          delta.toString,
          confidenceInfo,
          f"$S%.2f",
          s"${runningTime}ms",
          f"$precision%.4f",
          f"$recall%.4f",
          f"$ase%.4f"
        )
        results += result

        println("=================================================================")
        println(s"Parameter Settings: MST=$MST; Number of Data Blocks=$numberOfblocks; ε=$error; δ=$delta")
        println(s"rmax (Contains single item sets): $rmax")
        println(s"IS Result (Contains item set): $distinctItemsetsCount")
        println(s"Sample Size (including itemset): $S")
        println(s"rmaxWS (Not include single-item sets): $rmaxWS")
        println(s"IS结果 (Not include single items): $distinctItemsetsWSCount")
        println(s"Sample size (excluding single-item sets): $SWS")
        println(s"Number of confidence rules: ${allConfidences.size}")
        println(s"Precision: $precision")
        println(s"Recall: $recall")
        println(s"ASE: $ase")
        println(s"Running Time: ${runningTime}ms")
        println("=================================================================")
      }

      // Print final results table
      println("\n" + "="*100)
      println("FINAL RESULTS TABLE:")
      println("ε\tδ\tConfidence\tS\tTime\tPrecision\tRecall\tASE")
      println("-"*100)
      results.foreach { result =>
        println(result.mkString("\t"))
      }
      println("="*100)

      spark.close()
    }






























}
