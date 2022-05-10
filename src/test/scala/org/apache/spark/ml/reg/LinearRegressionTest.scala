package org.apache.spark.ml.reg

import com.google.common.io.Files
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val vectors: Seq[(Vector, Double)] = LinearRegressionTest._vectors

  "Estimator" should "calculate weights" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setPredictionCol("predictions")
      .setLabelCol("label")
      .setLearningRate(0.5)

    val model = estimator.fit(data)

    model.weights.leftSide

    model.weights(0) should be (1.0 +- 0.001)
    model.weights(1) should be (2.0 +- 0.001)
    model.bias should be(0.0 +- 0.001)
  }

  "Estimator" should "should produce functional model" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setPredictionCol("predictions")
      .setLabelCol("label")
      .setLearningRate(0.5)

    val model = estimator.fit(data)

    validateModel(model, model.transform(data))
  }

  private def validateModel(model: LinearRegressionModel, data: DataFrame) = {
    val vectors: Array[Double] = data.collect().map(_.getAs[Double]("predictions"))

    vectors.length should be(3)

    val delta = 0.001
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

    model.weights(0) should be (1.0 +- 0.001)
    model.weights(1) should be (2.0 +- 0.001)
    model.bias should be(0.0 +- 0.001)
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

    validateModel(model.stages(0).asInstanceOf[LinearRegressionModel], reRead.transform(data))
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
}
