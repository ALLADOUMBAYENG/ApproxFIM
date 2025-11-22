package org.apache.spark

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.RspContext.{NewRDDFunc, RspDatasetFunc, RspRDDFunc}
import org.apache.spark.mlutils.GO_Strategy.mergeFPGrowth
import org.apache.spark.mlutils.dataWrapper._
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext.SparkSessionFunc
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import smile.association.{ItemSet, fpgrowth}

import java.util.Date
import java.util.stream.Stream
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object ApproxFIMResult {
  def main(args: Array[String]): Unit = {

    // 获取 Hadoop 文件系统对象
    val hadoopConf = new Configuration()
    val hadoopFs = FileSystem.get(hadoopConf)
    // 指定要检查的目录路径
    val outputPath = new Path("output/data")
    // 检查目录是否存在
    if (hadoopFs.exists(outputPath)) {
      println(s"Directory $outputPath exists. Deleting...")
      // 删除目录
      hadoopFs.delete(outputPath, true)
      println(s"Directory $outputPath has been deleted.")
    } else {
      println(s"Directory $outputPath does not exist.")
    }
    // TODO 构建spark环境
    val sparkconf = new SparkConf().setAppName("FIMResult").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")
    val now_start: Date=new Date()
    val MST = 0.02
    //val numberofBlocks = 208
    //val frame1 = spark.read.csv("ApproxFIM/GenD"+MST)
    val frame1 = spark.read.csv("ApproxFIM/FIM1/TH20.02")

    val value1: RDD[(String, Int)] = frame1.rdd.map(a=>(a.getAs[String](0),a.getString(1).toInt))
    val Result1: collection.Map[String, Int] = value1.map(row => row._1 -> row._2).coalesce(1).collectAsMap()

    //val frame2 = spark.read.csv("ScaDistFIM/GenD"+MST)
    val frame2 = spark.read.csv("ScaDistFIM/TH20.02")
    val value2: RDD[(String, Int)] = frame2.rdd.map(a=>(a.getAs[String](0),a.getString(1).toInt))
    val Result2: collection.Map[String, Int] = value2.map(row => row._1 -> row._2).coalesce(1).collectAsMap()


    val now_end: Date=new Date()
    val runningTime = now_end.getTime - now_start.getTime
    print("===============================================" +
      "\n"+"Running time = "+ runningTime +"ms"
      +"\n"+"=================================================="+"\n")

//    val frame = spark.read.parquet("SparkFIMResult1/Kaggle"+MST)
//    val value: RDD[(String, Int)] = frame.rdd.map { row =>
//      // 使用 getAs 获取 items，并确保它是 Array[String] 类型
//      val items = row.getAs[mutable.WrappedArray[String]](0).mkString(",")
//      // 获取 freq 字段并转换为 Int
//      val freq = row.getLong(1).toInt
//      (items, freq)
//    }
//    val Result2: collection.Map[String, Int] = value.map(row => row._1 -> row._2).coalesce(1).collectAsMap()

    //不同数据需要更改此部分。

//    val dataRead = spark.rspRead.parquet("data/Kaggle")
////    val numberOfSampleBlocks = 29
////    val value3: RspRDD[Row] = dataRead.rdd.getSubPartitions(numberOfSampleBlocks).toRSP(numberOfSampleBlocks)
//    val numberOfData = dataRead.rdd.count().toInt

    val numberOfData = 1000000000
    GetInfo(Result1,Result2,numberOfData,10000)

//    GetInfo(Result1,Result2,numberOfData,dataRead.rdd.getNumPartitions)

    spark.close()
  }

  private def GetInfo(LOGO_Map: collection.Map[String, Int], Spark_Map: collection.Map[String, Int], TotalNum: Int, Block_Num: Int): Unit = {

    val SparkConvertedMap = Spark_Map.map {
      case (key, value) =>
        (key.replaceAll("[{}]", "").split(",").map(_.toInt).toSet, value)
    }
    val LOGOConvertedMap = LOGO_Map.map {
      case (key, value) =>
        (key.replaceAll("[{}]", "").split(",").map(_.toInt).toSet, value)
    }

    val Spark_FIM_List = SparkConvertedMap.keys.toList
    val LOGO_FIM_List = LOGOConvertedMap.keys.toList
    val TP_FIM = Spark_FIM_List.intersect(LOGO_FIM_List)

    // 设置统计信息
    val Real_Num: Int = Spark_Map.size
    val LOGO_Num: Int = LOGO_Map.size
    val TP: Int = TP_FIM.size

    println("Spark结果：" + Real_Num)
    println("LOGO结果：" + LOGO_Num)
    println("TP=" + TP)
    println("FP=" + (LOGO_Num - TP))
    println("FN=" + (Real_Num - TP))
    println("Precision：" + TP.toDouble / LOGO_Num)
    println("Recall：" + TP.toDouble / Real_Num)


    val table = ArrayBuffer[(Set[Int], Int)]()


    for (elem <- TP_FIM) {
      table.append((elem, math.abs(SparkConvertedMap.getOrElse(elem, 0) - LOGOConvertedMap.getOrElse(elem, 0) * Block_Num)))
    }

    val sum: Int = table.map(_._2).sum
    //println("Error误差：" + sum.toDouble / TP)
    println("Average Error Support(ASE)误差：" + sum.toDouble / TP / TotalNum)
  }
}
