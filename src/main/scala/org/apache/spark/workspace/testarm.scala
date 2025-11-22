package org.apache.spark.workspace

import org.apache.spark.ml.attribute.NominalAttribute
import smile.association.{ARM, AssociationRule, arm}

import java.util.stream
import java.util.stream.Collectors
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable

object testarm {
  //  def main(args: Array[String]): Unit = {
  //
  //    // 创建一个二维数组
  //    val numRows = 3
  //    val numCols = 4
  //    val twoDimensionalArray: Array[Array[Int]] = Array.ofDim[Int](numRows, numCols)
  //
  //    // 初始化二维数组的元素
  //    for (i <- 0 until numRows) {
  //      for (j <- 0 until numCols) {
  //        twoDimensionalArray(i)(j) = i * numCols + j
  //      }
  //    }
  //
  //    val value: stream.Stream[AssociationRule] = arm(1, 0.99, twoDimensionalArray)
  //
  //    println(value.limit(10))


  def main(args: Array[String]): Unit = {
    // 创建一个简单的事务数据集
    val numRows = 3
    val numCols = 4
    val dataset: Array[Array[Int]] = Array.ofDim[Int](numRows, numCols)

    // 初始化二维数组的元素
    for (i <- 0 until numRows) {
      for (j <- 0 until numCols) {
        dataset(i)(j) = i * numCols + j
      }
    }

    val value: stream.Stream[AssociationRule] = arm(0, 0.99, dataset)


    val arms: mutable.Buffer[AssociationRule] = value.collect(Collectors.toList[AssociationRule]).asScala

    arms.foreach(println)

    for (i <- 0 until numRows) {
      for (j <- 0 until numCols) {
        print(dataset(i)(j) + " ")
      }
      println()
    }



//    value.forEach { rule: AssociationRule =>
////      println(s"Rule: ${rule.getPremise.mkString(", ")} => ${rule.getConsequence.mkString(", ")}")
////      println(s"Support: ${rule.getSupport}, Confidence: ${rule.getConfidence}")
//      println(rule)
//      rule
//    }
  }


//    println(value.limit(10))




    // 执行ARM算法
    //      val minSupport = 0.5 // 最小支持度
    //      val minConfidence = 0.6 // 最小置信度
    //      val arm = new ARM(dataset, minSupport, minConfidence)
    //
    //      // 获取关联规则并输出
    //      val rules = arm.learnAssociationRules().asScala
    //      for (rule <- rules) {
    //        println(s"Rule: ${rule.getPremise.mkString(", ")} => ${rule.getConsequence.mkString(", ")}")
    //        println(s"Support: ${rule.getSupport}, Confidence: ${rule.getConfidence}")
    //        println()
    //      }
    //    }


        // 访问二维数组的元素



//  }
}
