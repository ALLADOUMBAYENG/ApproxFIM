package org.apache.spark

import org.apache.spark.mlutils.GO_Strategy.mergeFPGrowth
import org.apache.spark.mlutils.dataWrapper.Smile_Parquet_FPG
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext.{NewRDDFunc, RspDatasetFunc, RspRDDFunc, SparkSessionFunc}
import org.apache.spark.sql.{Row, RspDataset, SparkSession}
import smile.association.{ItemSet, fpgrowth}

import java.util.stream.Stream
import scala.collection.mutable.ArrayBuffer

object Test_20230627_testFPGdata2 {
  def main(args: Array[String]): Unit = {

    // TODO 构建spark环境
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")

    // 开始时间
    val startTime = System.nanoTime()

    val transactions: RspRDD[Array[Array[Int]]] = spark.rspRead.parquet("data/TH1_260W_10_26W")
      .dataWrapper(Smile_Parquet_FPG)

    val transactions2: RspRDD[Array[Array[Int]]] = spark.rspRead.parquet("data/TH1_260W_10_26W")
      .dataWrapper(Smile_Parquet_FPG)


    val value: RspDataset[Row] = spark.rspRead.parquet("data/TH1_260W_10_26W")
    println(value.rdd.count())


    println("------数据加载成功-------")

    val localTable: RspRDD[Stream[ItemSet]] = transactions.LO(trainDF =>
      fpgrowth((trainDF.length * 0.005).toInt, trainDF)
    )

    val localTable2: RspRDD[Stream[ItemSet]] = transactions2.LO(trainDF =>
      fpgrowth((trainDF.length * 0.005).toInt, trainDF)
    )

    println("第一种方式的RSP模型的数量：" + localTable.count())
    println("第二种方式的RSP模型的数量：" + localTable.partitions.length)

    println("-------------LO建模成功---------------")

    val goTable: RspRDD[(String, Int)] = localTable.GO(mergeFPGrowth)
    val goTable2: RspRDD[(String, Int)] = localTable2.GO(mergeFPGrowth)

    println("-------------GO集成成功---------------")

    //    goTable.foreach(println)
    goTable.count()

    // 结束时间
    val endTime = System.nanoTime()
    // 计算执行时间（以毫秒为单位）
    val executionTimeMs = (endTime - startTime) / 1000000

    //    goTable.saveAsTextFile("datas/T40I10D100K_result000")

    println("---------文件已保存-----------")

    // 打印执行时间
    println(s"程序执行时间：$executionTimeMs 毫秒")

    println(goTable.count())

    val Result1: collection.Map[String, Int] = goTable.map(row => row._1 -> row._2).coalesce(1).collectAsMap()
    val Result2: collection.Map[String, Int] = goTable2.map(row => row._1 -> row._2).coalesce(1).collectAsMap()

    GetInfo(Result1,Result2,value.rdd.count().toInt,transactions.getNumPartitions)

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
    println("TP值：" + TP)
    println("FP值：" + (Real_Num - TP))
    println("准确率：" + TP.toDouble / LOGO_Num)
    println("召回率：" + TP.toDouble / Real_Num)


    val table = ArrayBuffer[(Set[Int], Int)]()


    for (elem <- TP_FIM) {
      table.append((elem, math.abs(SparkConvertedMap.getOrElse(elem, 0) - LOGOConvertedMap.getOrElse(elem, 0) * Block_Num)))
    }

    val sum: Int = table.map(_._2).sum
    println("Error误差：" + sum.toDouble / TP)
    println("Support误差：" + sum.toDouble / TP / TotalNum)
  }


}
