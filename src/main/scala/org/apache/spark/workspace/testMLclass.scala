package org.apache.spark.workspace

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.logoML.associate_Rules.LOGO_FPG
import org.apache.spark.logoML.supervised.DecisionTree
import org.apache.spark.mlutils.GO_Strategy.mergeFPGrowth
import org.apache.spark.mlutils.dataWrapper.Smile_Parquet_FPG
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext.{RspDatasetFunc, RspRDDFunc, SparkSessionFunc}
import org.apache.spark.sql.SparkSession
import smile.association.ItemSet
import smile.data.DataFrame
import smile.data.formula.Formula

import java.util.stream

object testMLclass {

  def main(args: Array[String]): Unit = {

    // TODO 构建spark环境
    val conf = new SparkConf().setAppName("PatternMatchingRDD").setMaster("local[*]")
    val sc = new SparkContext(conf)

    println("------------环境配置成功---------------")

    val inputRDD = sc.parallelize(List(("apple", 3), ("banana", 4), ("grape", 5)))

    val transformedRDD = inputRDD.map {
      case (a, b) => (a, (b, 1))
    }

    val value = transformedRDD.reduceByKey {
      case ((freq1,votes1), (freq2,votes2)) => (freq1 + freq2, votes1 + votes2)
    }

    val value1 = value.filter{
      case (itemset,(freq,votes)) => votes >= 1 * 0.5
    }

    val result = value1.map {
      case (itemset, (freq, votes)) =>
        (itemset, freq / votes)
    }

    result.foreach(println)

    sc.stop()
  }
}
