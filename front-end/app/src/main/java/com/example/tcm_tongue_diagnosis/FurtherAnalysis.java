package com.example.tcm_tongue_diagnosis;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Patterns;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

public class FurtherAnalysis extends AppCompatActivity {

    private Button buttonSend;
    private Button buttonCancel;
    private ImageView furtherView;
    private EditText editEmail;
    private EditText editSymptom;
    private byte[] byteArray;
    private Bitmap bitmap;
    //private ImageView imageViewReceive;

     @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_further_analysis);

        furtherView = (ImageView) findViewById(R.id.imageViewFA);
        buttonSend = (Button) findViewById(R.id.button_send);
        buttonCancel = (Button) findViewById(R.id.button_cancel);
        editEmail = (EditText) findViewById(R.id.edit_email);
        editSymptom = (EditText) findViewById(R.id.edit_symptom);
        //imageViewReceive = (ImageView) findViewById(R.id.imageView3);

         byteArray = getIntent().getByteArrayExtra("Image");
         bitmap = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
         furtherView.setImageBitmap(bitmap);

         //Bitmap bitmap = (Bitmap) getIntent().getParcelableExtra("Bitmap");
         //imageViewReceive.setImageBitmap(bitmap);

         buttonSend.setOnClickListener(new View.OnClickListener() {
             @Override
             public void onClick(View v) {
                 try {
                     String emailCheck = editEmail.getText().toString();
                     String symptomCheck = editSymptom.getText().toString();
                     if(isEmailValid(emailCheck) == false && (isTextEmpty(symptomCheck) == true)){
                         editEmail.setBackgroundResource(R.drawable.button_border_error);
                         editSymptom.setBackgroundResource(R.drawable.button_border_error);
                     } else if(isEmailValid(emailCheck) == true && (isTextEmpty(symptomCheck) == true)) {
                         editEmail.setBackgroundResource(R.drawable.button_border);
                         editSymptom.setBackgroundResource(R.drawable.button_border_error);
                     } else if(isEmailValid(emailCheck) == false && (isTextEmpty(symptomCheck) == false)) {
                         editEmail.setBackgroundResource(R.drawable.button_border_error);
                         editSymptom.setBackgroundResource(R.drawable.button_border);
                     } else if(isEmailValid(emailCheck) == true && (isTextEmpty(symptomCheck) == false)){
                         editEmail.setBackgroundResource(R.drawable.button_border);
                         editSymptom.setBackgroundResource(R.drawable.button_border);
                         Intent intent = new Intent(FurtherAnalysis.this, Success.class);
                         startActivity(intent);
                     }
                 } catch (Exception e) {
                     e.printStackTrace();
                 }

             }
         });

         buttonCancel.setOnClickListener(new View.OnClickListener() {
             @Override
             public void onClick(View v) {
                 try {
                     Intent intent = new Intent(FurtherAnalysis.this, MainActivity.class);
                     startActivity(intent);
                 } catch (Exception e) {
                     e.printStackTrace();
                 }

             }
         });
    }
    boolean isEmailValid(CharSequence email) {
         return Patterns.EMAIL_ADDRESS.matcher(email).matches();
    }
    boolean isTextEmpty(String text) {
         return text.trim().equals("");
    }
}