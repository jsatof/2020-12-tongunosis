package com.example.tcm_tongue_diagnosis;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Timer;
import java.util.TimerTask;

public class AnalysisLoading extends AppCompatActivity {
    private static String TAG = "myApp";
    TextView loadingText;
    Animation rotateAnimation;
    ImageView logoLoading;
    String imageString;
    HttpURLConnection urlConnection;
    ByteArrayOutputStream byteOut;
    byte[] byteArray;
    byte[] byteArrayNew;

    @SuppressLint("WrongThread")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_analysis_loading);

        //new request().execute();
        new post().execute();
        /*Bitmap image = StringToBitMap(imageString);
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        image.compress(Bitmap.CompressFormat.PNG, 100, stream);
        byteArrayNew = stream.toByteArray();*/

        byteArray = getIntent().getByteArrayExtra("Image");

        logoLoading = (ImageView) findViewById(R.id.logo_loading);
        loadingText = (TextView) findViewById(R.id.loading);

        rotateAnimation();

        Timer timer = new Timer();
        timer.schedule(new TimerTask() {
            public void run() {
                try {
                    /*new request().execute();
                    new post().execute();
                    Britmap image = StingToBitMap(imageString);
                    ByteArrayOutputStream stream = new ByteArrayOutputStream();
                    image.compress(Bitmap.CompressFormat.PNG, 100, stream);
                    byteArrayNew = stream.toByteArray();*/

                    //byteArray = getIntent().getByteArrayExtra("Image");

                    Intent intent = new Intent(AnalysisLoading.this, Analysis.class);
                    intent.putExtra("Image", byteArray);
                    startActivity(intent);

                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }, 10000);

    }

    class request extends AsyncTask<String, Void, String> {
        StringBuffer sb = new StringBuffer();
        protected String doInBackground(String... params) {
            //get
            try {
                URL url = new URL("http://69.43.72.249/api/test");
                //temp new var
                urlConnection = (HttpURLConnection) url.openConnection();
                int code = urlConnection.getResponseCode();
                if(code != 200) {
                    throw new IOException("Server Code: " + code);
                }

                BufferedReader reader = new BufferedReader(new InputStreamReader(urlConnection.getInputStream()));
                String read;
                sb = new StringBuffer();
                while((read = reader.readLine()) != null) {
                    sb.append(read);
                }
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                if(urlConnection != null) {
                    urlConnection.disconnect();
                }
            }
            return sb.toString();
        }
        @Override
        protected void onPostExecute(String data) {
            Log.v(TAG, "DATA = " + data);
        }
    }

    class post extends AsyncTask<String, Void, String> {
        String line;
        @Override
        protected String doInBackground(String... params) {

            try {
                //url to website
                URL url = new URL("http://69.43.72.249/api/test");
                urlConnection = (HttpURLConnection) url.openConnection();
                urlConnection.setDoOutput(true);
                urlConnection.setDoInput(true);
                //type post
                urlConnection.setRequestMethod("POST");

                JsonObject postData = new JsonObject();
                JsonArray postDataArray = new JsonArray();

                //Bitmap bitmapPhoto = BitmapFactory.decodeFile("/drawable/tongue.jpg");

                /*ByteArrayOutputStream sendPhoto = new ByteArrayOutputStream();
                bitmapPhoto.compress(Bitmap.CompressFormat.JPEG, 100, sendPhoto);
                byte[] byteArrayPhoto = sendPhoto.toByteArray();

                Log.v(TAG, "PHOTO: " + byteArrayPhoto);

                String encodeImage = Base64.encodeToString(byteArrayPhoto, Base64.DEFAULT);*/

                //postData.addProperty("photo", String.valueOf(byteArray));
                // what we are sending
                postData.addProperty("photo", "hello");

                urlConnection.setRequestProperty("Content-Type", "application/json");

                //urlConnection.setChunkedStreamingMode(0);

                OutputStream out = new BufferedOutputStream(urlConnection.getOutputStream());

                BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out, "UTF-8"));
                //data is sent here
                writer.write(postData.toString());
                writer.flush();

                int code = urlConnection.getResponseCode();
                Log.v(TAG, "CODE: " + code);

                //any code other than success
                /*if(code != 200) {
                    throw new IOException("Server Code: " + code);
                }*/

                BufferedReader reader = new BufferedReader(new InputStreamReader(urlConnection.getInputStream()));
                Log.v(TAG, "Message: " + reader);

                while((line = reader.readLine()) != null) {
                    Log.v(TAG, line);
                }

            } catch(IOException e) {
                e.printStackTrace();
            } finally {
                if(urlConnection != null) {
                    urlConnection.disconnect();
                }

            }
            Bitmap serverImg = StringToBitMap(line);
            imageString = line;
            return line;
        }

        @Override
        protected void onPostExecute(String data) {
            Log.v(TAG, "DATA = " + data);
        }
    }

    public String BitMapToString(Bitmap bitmap){
        ByteArrayOutputStream output = new  ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG,100, output);
        byte [] byteArray = output.toByteArray();
        String temp=Base64.encodeToString(byteArray, Base64.DEFAULT);
        return temp;
    }

    public Bitmap StringToBitMap(String encodedString){
        try {
            byte [] encodeByte= Base64.decode(encodedString,Base64.DEFAULT);
            Bitmap bitmap = BitmapFactory.decodeByteArray(encodeByte, 0, encodeByte.length);
            return bitmap;
        } catch(Exception e) {
            e.getMessage();
            return null;
        }
    }

    private void writeStream(OutputStream out) {
        try {
            String output = "Hello World";
            out.write(output.getBytes());
            out.flush();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void rotateAnimation() {
        rotateAnimation = AnimationUtils.loadAnimation(this,R.anim.rotate_anim);
        logoLoading.startAnimation(rotateAnimation);
    }
}