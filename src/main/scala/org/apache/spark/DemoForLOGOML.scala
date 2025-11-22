package org.apache.spark

import org.apache.spark.RspContext.{RspDatasetFunc, RspRDDFunc, SparkSessionFunc}
import org.apache.spark.mlutils.GO_Strategy.mergeFPGrowth
import org.apache.spark.rsp.RspRDD
//import org.apache.spark.sql.RspContext.{RspRDDFunc, _}
import org.apache.spark.sql.{Row, RspDataset, SparkSession}
import smile.association.{ItemSet, fpgrowth}
import smile.classification.{DecisionTree, RandomForest, cart, randomForest}
import smile.data.DataFrame
import smile.data.formula.Formula

import java.util.stream.{Collectors, Stream}
import org.apache.spark.mlutils.dataWrapper._

object DemoForLOGOML {
  def main(args: Array[String]): Unit = {

    // TODO 构建spark环境
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")

    // TODO 数据加载并进行数据封装
    val inputDataFrame:RspRDD[DataFrame] = spark.rspRead.parquet("data/classification_100_2_0.54_4_32M.parquet")
      .getSubDataset(10)
      .dataWrapper(Smile_Parquet_Classification)
    println("------数据加载成功，并完成数据封装-------")


    // TODO LOGO阶段自定义训练器的执行方式
    def DT(trainFrame2: DataFrame): DecisionTree = {
      val tree: DecisionTree = cart(Formula.lhs("Y"), trainFrame2, maxDepth = 2, nodeSize = 2)
      tree
    }
    val modelRDD: RspRDD[DecisionTree] = inputDataFrame.LO(DT)

    // TODO LOGO阶段已有训练器的执行方式
    val modelRDD2: RspRDD[RandomForest] = inputDataFrame.LO(
      trainDF =>
        randomForest(Formula.lhs("Y"), trainDF, maxDepth = 2, nodeSize = 2, ntrees = 1)
    )

//    println(modelRDD.partitions.length)
//    println(modelRDD2.partitions.length)

    println("-------------LO建模成功---------------")

    println(modelRDD.foreach(println))
    println(modelRDD2.foreach(println))
    println("第一种方式的RSP模型的数量：" + modelRDD.partitions.length)
    println("第二种方式的RSP模型的数量：" + modelRDD2.count())
//    modelRDD2.foreach(x=> println("当前树" + x.trees().mkString("Array(", ", ", ")")))

//    // TODO 进行GO阶段的集成
//    //modelRDD.GO("classification")
//    println("------------GO集成成功---------------")
//    println("pass")
//
//
//    val transactions: RspRDD[Array[Array[Int]]]  = spark.rspRead.parquet("datas/transaction_dataset_30.parquet")
//      .dataWrapper(Smile_Parquet_FPG)
//
//    println("------数据加载成功-------")
//
//    val localTable: RspRDD[Stream[ItemSet]] = transactions.LO(trainDF =>
//      fpgrowth((trainDF.length * 0.21).toInt, trainDF)
//    )
//
//    println("第一种方式的RSP模型的数量：" + localTable.count())
//    println("第二种方式的RSP模型的数量：" + localTable.partitions.length)
//
//    println("-------------LO建模成功---------------")
//
//    val goTable: RspRDD[(String, Int)] = localTable.GO(mergeFPGrowth)
//
//    println("-------------GO集成成功---------------")
//
////    println(goTable.toDebugString)
//
//    goTable.foreach(println)




    spark.close()






  }
}
