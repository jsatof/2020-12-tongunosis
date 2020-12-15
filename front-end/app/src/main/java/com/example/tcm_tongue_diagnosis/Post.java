package com.example.tcm_tongue_diagnosis;

import com.google.gson.annotations.SerializedName;

public class Post {
    private Integer id;
    private String email;
    private String symptoms;
    private byte[] bitmap;

    @SerializedName("body")
    private String text;

    public Post(String email, String symptoms, byte[] bitmap, String text) {
        this.email = email;
        this.symptoms = symptoms;
        this.bitmap = bitmap;
        this.text = text;
    }

    public int getId() {
        return id;
    }

    public String getEmail() {
        return email;
    }

    public String getSymptoms() {
        return symptoms;
    }

    public byte[] getBitmap() {
        return bitmap;
    }
}
