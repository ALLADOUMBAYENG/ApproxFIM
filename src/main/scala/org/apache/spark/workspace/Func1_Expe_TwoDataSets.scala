package org.apache.spark.workspace

import org.apache.hadoop.io.DataOutputBuffer
import org.apache.spark.SparkConf
import org.apache.spark.mlutils.GO_Strategy.mergeFPGrowth
import org.apache.spark.mlutils.dataWrapper.Smile_Parquet_FPG
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext.{RspDatasetFunc, RspRDDFunc, SparkSessionFunc}
import org.apache.spark.sql.{Row, RspContext, RspDataset, SparkSession}
import smile.association.{ItemSet, fpgrowth}

import java.util.stream.Stream
import scala.collection.mutable.ArrayBuffer

object Func1_Expe_TwoDataSets {
  def main(args: Array[String]): Unit = {

    // TODO 构建spark环境
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")


    val trueFIM: RspRDD[(String, Int)] = Get_GoTable("datas/T10I4D100K_RSP_1blocks", 0.005, spark)

    // TODO 块数量变化 循环以计算结果
    cal_Info_Change_Block("T10I4D100K",
      3,21,2,0.005,trueFIM,spark)


    // TODO 支持度变化 循环以计算结果
    cal_Info_Change_Support("datas/T10I4D100K_RSP_1blocks","datas/T10I4D100K_RSP_5blocks",
      5,0.004,0.0058,0.0002,spark)


    spark.close()


  }

  // TODO 支持度变化 循环以计算结果
  def cal_Info_Change_Support(SparkDataPath: String,
                              LogoDataPath: String,
                              block_Num: Int,
                              start_Support: Double,
                              end_Support: Double,
                              gap: Double,
                              spark: SparkSession): Unit = {

    for (current_support <- start_Support to end_Support by gap) {

      val goTable_S: RspRDD[(String, Int)] = Get_GoTable(SparkDataPath, current_support, spark)
      val goTable_L: RspRDD[(String, Int)] = Get_GoTable(LogoDataPath, current_support, spark)

      GetInfo(goTable_L, goTable_S, 100000, block_Num)

      println("当前支持度：" + current_support)
    }

  }




  // TODO 块数量变化 循环以计算结果
  private def cal_Info_Change_Block(dataName: String,
                                    start_Block_Num: Int,
                                    end_Block_Num: Int,
                                    gap: Int,
                                    support: Double,
                                    trueFIM: RspRDD[(String, Int)],
                                    spark: SparkSession): Unit = {

    for (num <- start_Block_Num to end_Block_Num by gap) {

      val path = "datas/" + dataName + "_RSP_" + num + "blocks"
      val goTable: RspRDD[(String, Int)] = Get_GoTable(path, support, spark)
      GetInfo(goTable, trueFIM, 100000, num)

      println(path)
    }
  }

  private def Get_GoTable(path: String, support: Double, spark: SparkSession): RspRDD[(String, Int)] = {

    val transactions: RspRDD[Array[Array[Int]]] = spark.rspRead.parquet(path)
      .dataWrapper(Smile_Parquet_FPG)

    println("------数据加载成功-------")

    val localTable: RspRDD[Stream[ItemSet]] = transactions.LO(trainDF =>
      fpgrowth((trainDF.length * support).toInt, trainDF)
    )

    println("当前RSP模型的数量：" + localTable.count())

    val goTable: RspRDD[(String, Int)] = localTable.GO(mergeFPGrowth)

    println("-------------GO集成成功---------------")

    goTable

  }


  private def GetInfo(LOGO_GoTable: RspRDD[(String, Int)], True_GoTable: RspRDD[(String, Int)], TotalNum: Int, Block_Num: Int): Unit = {

    val Spark_Map: collection.Map[String, Int] = True_GoTable.map(row => row._1 -> row._2).coalesce(1).collectAsMap()
    val LOGO_Map: collection.Map[String, Int] = LOGO_GoTable.map(row => row._1 -> row._2).coalesce(1).collectAsMap()

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
