package com.namespace.imagesegmentation;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import java.util.Arrays;
import java.util.List;

import ai.fritz.core.Fritz;
import ai.fritz.vision.FritzVision;
import ai.fritz.vision.FritzVisionImage;
import ai.fritz.vision.FritzVisionModels;
import ai.fritz.vision.ModelVariant;
import ai.fritz.vision.imagesegmentation.FritzVisionSegmentationPredictor;
import ai.fritz.vision.imagesegmentation.FritzVisionSegmentationPredictorOptions;
import ai.fritz.vision.imagesegmentation.FritzVisionSegmentationResult;
import ai.fritz.vision.imagesegmentation.MaskClass;
import ai.fritz.vision.imagesegmentation.SegmentationOnDeviceModel;

public class MainActivity extends AppCompatActivity {
    Button buttonClick;
    ImageView imageView;
    FritzVisionSegmentationPredictorOptions options;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Fritz.configure(this, "YOUR_API_TOKEN");

        buttonClick = findViewById(R.id.buttonClick);
        imageView = findViewById(R.id.imageView);

    }
    public void segmentImages(View view){
        MaskClass[] maskClasses = {
                // The first class must be "None"
                new MaskClass("None", Color.TRANSPARENT),
                new MaskClass("cat", Color.RED),
                new MaskClass("dog", Color.BLUE),
                // ...
        };

        SegmentationOnDeviceModel onDeviceModel = new SegmentationOnDeviceModel(
                "file:///android_asset/CatDogSegmentationFast.tflite",
                "277782d7ef70455fafa550474877b426",
                2,
                maskClasses
        );

        Bitmap image = BitmapFactory.decodeResource(getResources(), R.drawable.cat);
        FritzVisionImage visionImage = FritzVisionImage.fromBitmap(image);
        FritzVisionSegmentationPredictor predictor = FritzVision.ImageSegmentation.getPredictor(onDeviceModel);

        FritzVisionSegmentationResult segmentationResult = predictor.predict(visionImage);

        options = new FritzVisionSegmentationPredictorOptions();
        options.confidenceThreshold = 0.1f;

        Log.i("info","The conf is "+segmentationResult.getMaskClassifications().toString());
        // Create a mask
        Bitmap petMask = segmentationResult.buildMultiClassMask(255, options.confidenceThreshold, options.confidenceThreshold);
        Bitmap imageWithMask = visionImage.overlay(petMask);
        imageView.setImageBitmap(imageWithMask);

    }
}