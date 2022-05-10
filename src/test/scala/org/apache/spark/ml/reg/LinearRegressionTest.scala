package org.apache.spark.ml.reg

import com.google.common.io.Files
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.0001
  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val noisyData: DataFrame = LinearRegressionTest._noisyData

  "Estimator" should "calculate weights" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setPredictionCol("predictions")
      .setLabelCol("label")
      .setLearningRate(0.5)

    val model = estimator.fit(data)

    model.weights.leftSide

    model.weights(0) should be (1.0 +- delta)
    model.weights(1) should be (2.0 +- delta)
    model.bias should be(0.0 +- delta)
  }

  "Estimator" should "calculate weights for noisy data" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setPredictionCol("predictions")
      .setLabelCol("label")
      .setLearningRate(0.5)
//      .setIterations(5000)

    val model = estimator.fit(noisyData)

    model.weights.leftSide

    val delta = 0.01
    model.weights(0) should be (0.5 +- delta)
    model.weights(1) should be (-1.0 +- delta)
    model.weights(2) should be (0.2 +- delta)
    model.bias should be(-0.1 +- delta)
  }

  "Estimator" should "should produce functional model" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setPredictionCol("predictions")
      .setLabelCol("label")
      .setLearningRate(0.5)

    val model = estimator.fit(data)

    validatePredictions(model.transform(data))
  }

  private def validatePredictions(data: DataFrame) = {
    val vectors: Array[Double] = data.collect().map(_.getAs[Double]("predictions"))

    vectors.length should be(3)

    vectors(0) should be(1.0 +- delta)
    vectors(1) should be(2.0 +- delta)
    vectors(2) should be(3.0 +- delta)
  }

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setLabelCol("label")
        .setLearningRate(0.5)
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    model.weights(0) should be (1.0 +- delta)
    model.weights(1) should be (2.0 +- delta)
    model.bias should be(0.0 +- delta)
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setPredictionCol("predictions")
        .setLabelCol("label")
        .setLearningRate(0.5)
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    validatePredictions(reRead.transform(data))
  }
}

object LinearRegressionTest extends WithSpark {

  lazy val _vectors: Seq[(Vector, Double)] = Seq(
    Tuple2(Vectors.dense(1, 0), 1.0),
    Tuple2(Vectors.dense(0, 1), 2.0),
    Tuple2(Vectors.dense(1, 1), 3.0),
  )

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _vectors.toDF("features", "label")
  }

  lazy val _noisyVectors: Seq[(Vector, Double)] = Seq(
    Tuple2(Vectors.dense(-0.17, -0.4, -0.31), 0.163),
    Tuple2(Vectors.dense(-0.61, -0.76, -0.28), 0.3073),
    Tuple2(Vectors.dense(0.79, 0.3, 0.07), 0.0177),
    Tuple2(Vectors.dense(0.83, 0.77, -0.95), -0.6388),
    Tuple2(Vectors.dense(0.86, 0.25, 0.07), 0.0943),
    Tuple2(Vectors.dense(0.35, 0.37, 0.8), -0.1257),
    Tuple2(Vectors.dense(0.23, 0.93, -0.62), -1.0366),
    Tuple2(Vectors.dense(0.14, 0.69, 0.81), -0.5577),
    Tuple2(Vectors.dense(0.82, 0.2, -0.81), -0.0429),
    Tuple2(Vectors.dense(-0.54, 0.19, 0.65), -0.4224)
  )

  lazy val _noisyWeights: Array[Double] = Array(0.5, -1, 0.2, -0.1)

  lazy val _noisyData: DataFrame = {
    import sqlc.implicits._
    _noisyVectors.toDF("features", "label")
  }
}
