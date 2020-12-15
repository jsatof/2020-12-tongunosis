package com.example.tcm_tongue_diagnosis;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.io.ByteArrayOutputStream;

public class PhotoConfirm extends AppCompatActivity {

    private Button buttonConfirm;
    private Button buttonRetake;
    private ImageView photoImage;
    public static final int CAMERA_REQUEST_CODE = 102;
    private byte[] byteArray;
    private Bitmap bitmap;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_photo_confirm);
        buttonConfirm = findViewById(R.id.button_confirm);
        buttonRetake = findViewById(R.id.button_retake);
        photoImage = findViewById(R.id.imageView);

        byteArray = getIntent().getByteArrayExtra("Image");
        bitmap = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
        photoImage.setImageBitmap(bitmap);

        /*String filepath = intent.getStringExtra("filepath");
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 8;
        filepath = filepath.replace("file://", "");
        File imgFile = new File(filepath);*/
        /*if(imgFile.exists()) {
            Bitmap bitmap = new BitmapFactory().decodeFile(imgFile.getAbsolutePath(), options);
            photoImage.setImageBitmap(bitmap);
        }*/

        /*Bundle extras = getIntent().getExtras();
        byte[] byteArray = extras.getByteArray("picture");

        Bitmap bmp = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
        ImageView image = (ImageView) findViewById(R.id.imageView);

        image.setImageBitmap(bmp);*/

        /*Intent receiveIntent = getIntent();
        String filepath = receiveIntent.getStringExtra("filepath");
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 8; // down sizing image as it throws OutOfMemory  Exception for larger images
        filepath = filepath.replace("file://", ""); // remove to avoid  BitmapFactory.decodeFile return null
        File imgFile = new File(filepath);

        if (imgFile.exists()) {
            Bitmap bitmap = BitmapFactory.decodeFile(imgFile.getAbsolutePath(),  options);
            photoImage.setImageBitmap(bitmap);
        }*/

        buttonConfirm.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
            try {
                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
                byte[] byteArray2 = stream.toByteArray();
                Intent intent = new Intent(PhotoConfirm.this, AnalysisLoading.class);
                intent.putExtra("Image", byteArray2);
                startActivity(intent);
            }
            catch(Exception e) {
                e.printStackTrace();
                }
            }
        });

        buttonRetake.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openCamera();
            }
        });
    }

    private void openCamera() {
        Intent camera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(camera, CAMERA_REQUEST_CODE);
    }

    @SuppressLint("MissingSuperCall")
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(requestCode == CAMERA_REQUEST_CODE) {
            Bitmap image = (Bitmap) data.getExtras().get("data");
            //compressed image to server?
            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            image.compress(Bitmap.CompressFormat.PNG, 100, stream);
            byteArray = stream.toByteArray();

            Intent intent = new Intent(this, PhotoConfirm.class);
            intent.putExtra("Image", byteArray);

            //GETTER using Retrofit

            /*Retrofit retrofit = new Retrofit.Builder()
                    .baseUrl("http://69.43.72.249/api/")
                    .addConverterFactory(GsonConverterFactory.create())
                    .build();

            jsonApi = retrofit.create(JsonAPI.class);

            createPost();

            Call<List<Post>> call = jsonApi.getPosts();

            call.enqueue(new Callback<List<Post>>() {
                @Override
                public void onResponse(Call<List<Post>> call, Response<List<Post>> response) {
                    if(!response.isSuccessful()) {

                    }

                    List<Post> posts = response.body();
                    for(Post post : posts) {
                        String content = "";
                        content += "ID: " + post.getId() + "\n";
                    }
                }

                @Override
                public void onFailure(Call<List<Post>> call, Throwable t) {

                }
            });*/

            //Intent intent = new Intent(this, PhotoConfirm.class);
            //intent.putExtra("Image", image);
            startActivity(intent);
            //selectedImage.setImageBitmap(image);
        }
    }
}