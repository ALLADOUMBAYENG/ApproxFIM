package org.apache.spark.workspace

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mlutils.dataWrapper.Smile_Parquet_FPG
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext.{RspDatasetFunc, SparkSessionFunc}
import org.apache.spark.sql.{Row, SparkSession}

import scala.collection.mutable
import scala.util.Random

object generatePriceTable {
  def main(args: Array[String]): Unit = {

    // TODO 构建spark环境
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val sc = new SparkContext(sparkconf)
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")

    // TODO 数据加载并进行数据封装
    val inputDataFrame = spark.rspRead.parquet("datas/transaction_dataset_30.parquet")

//    val value: RDD[String] = inputDataFrame.rdd.map(
//      row => {
//        row.getAs[String]("items")
//      }
//    )

    println(inputDataFrame.rdd.count())


    val value = inputDataFrame.rdd.map(_.get(0).asInstanceOf[mutable.WrappedArray[Int]].toArray.toSet)

    var ints = value.reduce(_ ++ _).toArray

//    println(ints.length)
//
//    ints = ints :+ 10246
//
//    ints = ints :+ 99
//
//    println(ints.length)




    val array: Array[String] = Array("food", "egg", "car", "fish", "tools")
    // 创建随机数生成器
    val random: Random = new Random()
    // 定义函数来获取随机元素
    def getRandomElement(array: Array[String]): String = {
      val randomIndex: Int = random.nextInt(array.length)
      array(randomIndex)
    }
    // 调用函数来获取随机元素
    val randomElement: String = getRandomElement(array)
    // 打印随机元素
    println(randomElement)


    // 定义函数来获取随机浮点数
    def getRandomFloat(range:Int): Double = {
      val randomDouble: Double = random.nextDouble() // 生成介于 0 到 1 之间的随机浮点数
      val scaledFloat: Double = randomDouble * range // 缩放到 1 到 1000 之间
      BigDecimal(scaledFloat).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble // 保留两位小数
    }
    // 调用函数来获取随机浮点数
    val randomFloat: Double = getRandomFloat(1000)
    // 打印随机浮点数
    println(randomFloat)


    for (goods <- ints) {
      println("{\"name\":" + " \"" + goods.toString + "\"," + "\"price\": " + getRandomFloat(1000) + "," + "\"Type\": \"" + getRandomElement(array) + "\"}")
    }

    spark.close()

  }
}
