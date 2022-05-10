package com.example.covid_x_ray_image_prediction;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.covid_x_ray_image_prediction.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    Button upload, predict;
    ImageView imageView;

    TextView result;
    TextView pcr;

    int img_size = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        upload = findViewById(R.id.button);
        imageView = findViewById(R.id.imageView);
        result = findViewById(R.id.textView);
        pcr = findViewById(R.id.textView2);

        upload.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                //Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(intent, 1);
            }
        });
    }
    private void classifyImage(Bitmap image) {
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * img_size * img_size * 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            inputFeature0.loadBuffer(byteBuffer);

            int[] intValues = new int[img_size * img_size];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;

            for(int i = 0; i < img_size; i++){
                for(int j = 0; j < img_size; j++){
                    int val = intValues[pixel++];   //RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                }
            }

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidence = outputFeature0.getFloatArray();

            int maxPos = 0;
            float maxConfidence = 0;
            float sum = 0;
            for(int i = 0; i < confidence.length; i++){
                sum += confidence[i];
                if(confidence[i] > maxConfidence){
                    maxConfidence = confidence[i];
                    maxPos = i;
                }
            }
            String[] classes = {"Covid", "Normal"};
            StringBuilder Builder = new StringBuilder();
            //percentage assessment
            if(confidence[0] > 0 && confidence[1] > 0){
                Builder.append(classes[0])
                        .append(":")
                        .append(confidence[0] / sum * 100)
                        .append("%")
                        .append("\n")
                        .append(classes[1])
                        .append(":")
                        .append(confidence[1] / sum * 100)
                        .append("%")
                        .append("\n");
            }
            else if(confidence[0] < 0 && confidence[1] < 0){
                Builder.append(classes[0])
                        .append(":")
                        .append(confidence[1] / sum * -100)
                        .append("%")
                        .append("\n")
                        .append(classes[1])
                        .append(":")
                        .append(confidence[0] / sum * -100)
                        .append("%")
                        .append("\n");
            }
            else if(confidence[0] > 0 && confidence[1] < 0){
                Builder.append(classes[0])
                        .append(":")
                        .append("100")
                        .append("%")
                        .append("\n")
                        .append(classes[1])
                        .append(":")
                        .append("0")
                        .append("%")
                        .append("\n");
            }
            else if(confidence[0] < 0 && confidence[1] > 0){
                Builder.append(classes[0])
                        .append(":")
                        .append("0")
                        .append("%")
                        .append("\n")
                        .append(classes[1])
                        .append(":")
                        .append("100")
                        .append("%")
                        .append("\n");
            }

            result.setText(Builder);

            if(maxPos == 0){
                pcr.setTextColor(Color.RED);
                pcr.setText(classes[maxPos]);
            }
            else{
                pcr.setTextColor(Color.GREEN);
                pcr.setText(classes[maxPos]);
            }




            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }



    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == 1) {
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, img_size, img_size, false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }


}