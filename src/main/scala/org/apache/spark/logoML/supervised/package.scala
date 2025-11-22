package org.apache.spark.logoML

import smile.association.{ItemSet, fpgrowth}
import smile.base.cart.SplitRule
import smile.classification.{RandomForest, cart, randomForest}
import smile.data.DataFrame
import smile.data.formula.Formula
import smile.util.time

import java.util.stream.LongStream
import java.util.stream.Stream


package object supervised {

  def logoDecisionTree(formula: Formula, data: DataFrame, splitRule: SplitRule = SplitRule.GINI, maxDepth: Int = 20, maxNodes: Int = 0, nodeSize: Int = 5): smile.classification.DecisionTree = time("LOGO Decision Tree"){
    cart(formula, data, splitRule, maxDepth, maxNodes, nodeSize)
  }

  def logoRandomForest(formula: Formula, data: DataFrame, ntrees: Int = 500, mtry: Int = 0,
                       splitRule: SplitRule = SplitRule.GINI, maxDepth: Int = 20, maxNodes: Int = 500,
                       nodeSize: Int = 1, subsample: Double = 1.0, classWeight: Array[Int] = null,
                       seeds: LongStream = null): RandomForest = time("LOGO Random Forest") {

    randomForest(formula,data: DataFrame,ntrees,mtry,splitRule, maxDepth, maxNodes, nodeSize, subsample, classWeight, seeds)
  }

//  def logoFPGrowth(minSupport: Int, itemsets: Array[Array[Int]]): Stream[ItemSet] = {
//    fpgrowth(minSupport,itemsets)
//  }



}
