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
      .setPredictionCol("prediction")
      .setLabelCol("label")

    val model = estimator.fit(data)
  }

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setLabelCol("label")
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

//    model.means(0) should be(vectors.map(_(0)).sum / vectors.length +- delta)
//    model.means(1) should be(vectors.map(_(1)).sum / vectors.length +- delta)
//
//    validateModel(model, model.transform(data))
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setPredictionCol("predictions")
        .setLabelCol("label")
    ))

    val model = pipeline.fit(data)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)
  }
}

object LinearRegressionTest extends WithSpark {

  lazy val _vectors = Seq(
    Tuple2(Vectors.dense(13.5, 12), 1.0),
    Tuple2(Vectors.dense(-1, 0), 2.0),
  )

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _vectors.toDF("features", "label")
  }
}