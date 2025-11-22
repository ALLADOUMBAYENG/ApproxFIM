package org.apache.spark.logoML.supervised

import org.apache.spark.logo.logoML
import org.apache.spark.mlutils.dataWrapper.{Smile_Parquet_Classification, Smile_Parquet_FPG}
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext.{RspDatasetFunc, RspRDDFunc}
import org.apache.spark.sql.{Row, RspDataset}
import smile.base.cart.SplitRule
import smile.classification
import smile.classification.cart
import smile.data.DataFrame
import smile.data.formula.Formula

import scala.reflect.ClassTag

class DecisionTree[T: ClassTag](rdd: RspRDD[DataFrame]) extends supervisedAl {

  val trainData: smile.data.DataFrame = null

  def logoDecisionTree(formula: Formula, data: DataFrame, splitRule: SplitRule = SplitRule.GINI, maxDepth: Int = 20, maxNodes: Int = 0, nodeSize: Int = 5): smile.classification.DecisionTree = {
    cart(formula, data, splitRule, maxDepth, maxNodes, nodeSize)
  }

  def trainer(formula: Formula, splitRule: SplitRule = SplitRule.GINI, maxDepth: Int = 20, maxNodes: Int = 0, nodeSize: Int = 5): RspRDD[classification.DecisionTree] = {

    this.rdd.LO(trainFrame => cart(Formula.lhs("Y"), trainFrame, maxDepth = 2, nodeSize = 2))

  }

}

object DecisionTree {
  def logoDecisionTree(formula: Formula, data: DataFrame, splitRule: SplitRule = SplitRule.GINI, maxDepth: Int = 20, maxNodes: Int = 0, nodeSize: Int = 5): smile.classification.DecisionTree = {
    cart(formula, data, splitRule, maxDepth, maxNodes, nodeSize)
  }
}
