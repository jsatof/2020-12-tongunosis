package com.example.tcm_tongue_diagnosis;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.ByteArrayOutputStream;

public class Analysis extends AppCompatActivity {

    private Button buttonHome;
    private Button buttonAnalysis;
    private ImageView photoView;
    private byte[] byteArray;
    private Bitmap bitmap;
    //private ImageView imageViewTransfer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_analysis);

        photoView = (ImageView) findViewById(R.id.imageViewA);
        buttonHome = (Button) findViewById(R.id.button_home);
        buttonAnalysis = (Button) findViewById(R.id.button_further_analysis);
        //imageViewTransfer = (ImageView) findViewById(R.id.imageView4);

        byteArray = getIntent().getByteArrayExtra("Image");
        bitmap = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
        photoView.setImageBitmap(bitmap);

        buttonHome.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    //imageViewTransfer.setDrawingCacheEnabled(true);
                    //Bitmap b = imageViewTransfer.getDrawingCache();
                    Intent intent = new Intent(Analysis.this, MainActivity.class);
                    //intent.putExtra("Bitmap", b);
                    startActivity(intent);
                } catch (Exception e) {
                    e.printStackTrace();
                }

            }
        });

        buttonAnalysis.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    ByteArrayOutputStream stream = new ByteArrayOutputStream();
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
                    byte[] byteArray2 = stream.toByteArray();
                    Intent intent = new Intent(Analysis.this, FurtherAnalysis.class);
                    intent.putExtra("Image", byteArray2);
                    startActivity(intent);
                } catch (Exception e) {
                    e.printStackTrace();
                }

            }
        });
    }
}