package org.apache.spark.workspace

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.fpm.FPGrowth

import scala.util.Random
import java.io.File
import java.util.Date
import scala.collection.mutable.ArrayBuffer

object Confidence2 {
   def main(args: Array[String]): Unit = {
     // Spark Session start
     val spark = SparkSession.builder()
       .appName("Frequent Itemset Mining")
       .master("local[*]")
       .config("spark.sql.adaptive.enabled", "true")
       .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
       .getOrCreate()

     //val now_start: Date=new Date()
     //val epsilon = 0.01
     //val delta = 0.05

     // Set log level to avoid verbose output
     // spark.sparkContext.setLogLevel("WARN")

     import spark.implicits._

     try {

     val directory = "data/TH2_260W_260_26W"
     val numFilesToProcess = 10 // number of random files that we want to analyze
     val dataFiles = getRandomFiles(directory, numFilesToProcess)
     processDataBlocks(dataFiles, spark)
   } catch {case e: Exception =>
     println(s"Error occured: ${e.getMessage}")
     e.printStackTrace()
     } finally {
       spark.stop()
     }
   }

    def readData(filePath: String, spark: SparkSession): DataFrame = {
      spark.read.parquet(filePath)
    }

    def calculateSupportThreshold(data: DataFrame): Double = {
      0.005  // min_support
    }

    def runFPGrowth(data: DataFrame, minSupport: Double): DataFrame = {
      val fpGrowth = new FPGrowth()
        .setItemsCol("items")
        .setMinSupport(minSupport)

      val model = fpGrowth.fit(data)
      model.freqItemsets
    }

    def calculateRmax(frequentItemsets: DataFrame, itemsetsCount: Long): Double = {
      if (frequentItemsets.count() > 0) {
        val maxSupport = frequentItemsets.agg(max("freq")).collect()(0).getLong(0)
        maxSupport.toDouble / itemsetsCount
      } else {
        0.0
      }
    }

    // return a list of random chosen files from directory
    def getRandomFiles(directory: String, numFiles: Int): Array[String] = {
      val dir = new File(directory)
      val allFiles = dir.listFiles()
        .filter(_.getName.endsWith(".parquet"))
        .map(_.getAbsolutePath)

      if (allFiles.length < numFiles) {
        allFiles
      } else {
        Random.shuffle(allFiles.toList).take(numFiles).toArray
      }
    }

    def processDataBlocks(dataFiles: Array[String], spark: SparkSession): Unit = {

      val rmaxValues = ArrayBuffer[Double]()
      var mergedItemsets: Option[DataFrame] = None
      val randomlyChosenDataBlocks = ArrayBuffer[String]()

      for (filePath <- dataFiles) {
        val data = readData(filePath, spark)
        val totalTransactions = data.count()
        val mst = calculateSupportThreshold(data)
        val frequentItemsets = runFPGrowth(data, mst)

        val totalItemsets = frequentItemsets.count()
        val rmax = calculateRmax(frequentItemsets, totalTransactions)

        // data block frame to show
        val frequentItemsetsData = frequentItemsets
          .withColumn("support", col("freq") / totalTransactions)

        rmaxValues += rmax
        randomlyChosenDataBlocks += filePath

        // sort data block
        val first20FrequentItemsetsSorted = frequentItemsetsData.orderBy(desc("support"))

        println(s"Data block: $filePath")
        println(s"Rmax for this block: $rmax")
        println(s"Total number of frequent itemsets: $totalItemsets")

        first20FrequentItemsetsSorted.show()

        // union all data blocks
        mergedItemsets = mergedItemsets match {
          case None => Some(frequentItemsetsData)
          case Some(existing) =>
            Some(existing.union(frequentItemsetsData).dropDuplicates("items"))
        }
      }

      println("RESULT ------------------------------------------------------")
      // rmax result calculation
      val rmaxFinal = {
        val sorted = rmaxValues.sorted
        val index = (0.95 * (rmaxValues.length - 1)).toInt
        sorted(index)
      }

      println("Randomly chosen data blocks:")
      randomlyChosenDataBlocks.foreach(println)
      println("List of maximum support values (Rmaxs) from data blocks:")
      rmaxValues.foreach(println)
      println(s"Rmax final from data blocks with quantile: $rmaxFinal")

      // Process merged itemsets
      mergedItemsets.foreach { merged =>
        val totalMergedItemsets = merged.count()

        println("-------------------------------------------------------------")
        println("Merged and deduplicated Data")
        println(s"Total number of frequent itemsets: $totalMergedItemsets")
        merged.orderBy(desc("support")).show()
      }
    }
  }