package com.example.tcm_tongue_diagnosis;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.util.HashMap;
import java.util.Map;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainActivity extends AppCompatActivity {
    private Button buttonBegin;
    private File photoFile;
    String currentImagePath = null;
    private final Context mContext = this;
    private Uri mFileUri;
    static final int CAMERA_PIC_REQUEST = 1;
    private byte[] byteArray;
    private JsonAPI jsonApi;
    public static final int CAMERA_PERM_CODE = 101;
    public static final int CAMERA_REQUEST_CODE = 102;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        buttonBegin = findViewById(R.id.button_begin);
        //Box box = new Box(this);
        buttonBegin.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                askCameraPermissions();

                    //File outputFile = new File(Environment.getExternalStorageDirectory() + "/myOutput.jpg");
                    //Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    //mFileUri = getOutputMediaFileUri(1);
                    //cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, outputFile.getAbsolutePath());
                    //startActivityForResult(cameraIntent, CAMERA_PIC_REQUEST);
                    //onActivityResult(CAMERA_PIC_REQUEST, RESULT_OK, cameraIntent);



                    /*File imageFolder = new File(Environment.getExternalStorageDirectory(), "YourFolderName");
                    if( !imageFolder.exists()) {
                        imageFolder.mkdir();
                    }
                    File imageFile = new File(imageFolder, "user.jpg");
                    Uri uriSavedImage = Uri.fromFile(image);
                    cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, uriSavedImage);
                    cameraIntent.putExtra(MediaStore.EXTRA_SCREEN_ORIENTATION, ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
                    startActivityForResult(cameraIntent, CAMERA_PIC_REQUEST);*/

                    /*Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    photoFile = getOutputMediaFile(1);
                    intent.putExtra(MediaStore.EXTRA_OUTPUT, photoFile);
                    startActivityForResult(intent, 100);*/
                    //startActivityForResult(intent, 0);
                    //ByteArrayOutputStream stream = new ByteArrayOutputStream();
                    //Intent intent = new Intent();
                    //intent.setAction(MediaStore.ACTION_IMAGE_CAPTURE);
                    //startActivity(intent);
            }
        });

    }

    private void askCameraPermissions() {
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, CAMERA_PERM_CODE);
        } else {
            openCamera();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if(requestCode == CAMERA_PERM_CODE) {
            if(grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera();
            } else {
                Toast.makeText(this, "This application requires camera access to function properly.", Toast.LENGTH_LONG).show();
            }
        }
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

    private void createPost() {
        Post post = new Post("rozanomax@gmail.com", "stomach pain", byteArray, "New Text");

        Map<String, String> fields = new HashMap<>();
        fields.put("email", "rozanomax@gmail.com");
        fields.put("symptoms", "stomach pain");

        Call<Post> call = jsonApi.createPost("rozanomax@gmail.com", "stomach pain", byteArray, "New Text");

        call.enqueue(new Callback<Post>() {
            @Override
            public void onResponse(Call<Post> call, Response<Post> response) {
                if(!response.isSuccessful()) {
                    //textViewResult.setText("Code: " + response.code());
                    return;
                }
                Post postResponse = response.body();


            }

            @Override
            public void onFailure(Call<Post> call, Throwable t) {

            }
        });
    }

    /*@Override
    protected void onActivityResult(int requestCode, int resultCode, Intent imageReturnedIntent) {
        super.onActivityResult(requestCode, resultCode, imageReturnedIntent);

        if (resultCode == RESULT_OK) {
            if (mFileUri != null) {
                String mFilePath = mFileUri.toString();
                if (mFilePath != null) {
                    Intent intent = new Intent(mContext, PhotoConfirm.class);
                    intent.putExtra("filepath", mFilePath);
                    startActivity(intent);
                }
            }
        }

        // refresh phone's folder content
        //sendBroadcast(new Intent(Intent.ACTION_MEDIA_MOUNTED, Uri.parse("file://" + Environment.getExternalStorageDirectory())));
    }*/

    /*private File getImageFile() throws IOException {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageName = "jpg_" + timeStamp + "";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);

        File imageFile = File.createTempFile(imageName, ".jpg", storageDir);
        currentImagePath = imageFile.getAbsolutePath();
        return imageFile;
    }*/

    /*private void captureImage() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

        mFileUri = getOutputMediaFileUri(1);

        intent.putExtra(MediaStore.EXTRA_OUTPUT, mFileUri);

        // start the image capture Intent
        startActivityForResult(intent, 100);
    }*/


    /*private Uri getOutputMediaFileUri(int type) {
        return Uri.fromFile(getOutputMediaFile(type));
    }*/

    /*private static File getOutputMediaFile(int type) {
        // External sdcard location
        File mediaStorageDir = new File(Environment.getExternalStorageDirectory(), "DCIM/Camera");

        // Create the storage directory if it does not exist
        if (!mediaStorageDir.exists()) {
            if (!mediaStorageDir.mkdirs()) {
                return null;
            }
        }

        // Create a media file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        File mediaFile;
        if (type == 1) {
            mediaFile = new File(mediaStorageDir.getPath() + File.separator + "IMG_" + timeStamp + ".jpg");
        } else {
            return null;
        }

        return mediaFile;
    }*/

    /*@Override
    protected void onActivityResult(int requestCode, int resultCode, Intent imageReturnedIntent) {
        super.onActivityResult(requestCode, resultCode, imageReturnedIntent);

            if (photoFile != null) {
                String mFilePath = photoFile.toString();
                if (mFilePath != null) {
                    Intent intent = new Intent(mContext, PhotoConfirm.class);
                    intent.putExtra("filepath", mFilePath);
                    startActivity(intent);
                }
            }
        }*/

    /*private static File getOutputMediaFile(int type) {
        File mediaStorageDir = new File(Environment.getExternalStorageDirectory(), "DCIM/Camera");
        if(!mediaStorageDir.exists()) {
            if(!mediaStorageDir.mkdirs()) {
                return null;
            }
        }
        String timeStamp = new SimpleDateFormat("yyyy.MM.dd_HH:mm:ss", Locale.getDefault()).format(new Date());
        File mediaFile;
        if(type == 1) {
            mediaFile = new File(mediaStorageDir.getPath() + File.separator + "IMG_" + timeStamp + ".jpg");
        }
        else {
            return null;
        }
        return mediaFile;
    }*/

    /*public class Box extends View {
        private Paint paint = new Paint();
        Box(Context context) {
            super(context);
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);

            paint.setStyle(Paint.Style.STROKE);
            paint.setColor(Color.YELLOW);
            paint.setStrokeWidth(10);

            int x0 = canvas.getWidth()/2;
            int y0 = canvas.getHeight()/2;
            int dx = canvas.getHeight()/3;
            int dy = canvas.getHeight()/3;

            canvas.drawRect(x0-dx, y0-dy, x0+dx, y0+dy, paint);
        }


    }*/

}