package com.titan.text_alchemy.user.utils;

import org.springframework.stereotype.Component;

import java.security.SecureRandom;

public class UserUtils {

    public static String generateOTP(int digit) {

        if (digit < 4 || digit > 10) {
            throw new IllegalArgumentException("Digit must be between 4 and 10");
        }

        String numbers = "0123456789";
        SecureRandom random = new SecureRandom();
        StringBuilder otp = new StringBuilder();

        for (int i = 0; i < digit; i++) {
            otp.append(numbers.charAt(random.nextInt(numbers.length())));
        }

        return otp.toString();

    }
}
