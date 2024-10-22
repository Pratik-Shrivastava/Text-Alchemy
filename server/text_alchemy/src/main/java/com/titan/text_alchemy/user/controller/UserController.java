package com.titan.text_alchemy.user.controller;

import com.titan.text_alchemy.common.model.GeneralResponse;
import com.titan.text_alchemy.common.service.EmailService;
import com.titan.text_alchemy.common.service.RedisService;
import com.titan.text_alchemy.user.model.User;
import com.titan.text_alchemy.user.service.UserService;
import com.titan.text_alchemy.user.utils.UserUtils;
import jakarta.mail.MessagingException;
import lombok.AllArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@AllArgsConstructor
@RequestMapping("/user")
public class UserController {

    @Autowired
    private final UserService userService;

    @Autowired
    private final EmailService emailService;

    @Autowired
    private final RedisService redisService;

    @PostMapping("/generate-otp")
    public ResponseEntity<GeneralResponse> generateOtp(
            @RequestParam String email
    )
    {

        User user = this.userService.getUserByEmail(email);

        if(user == null) {
            return new ResponseEntity<>(
                    new GeneralResponse(
                            false,
                            400,
                            "User does not exist!",
                            null
                    ),
                    HttpStatus.BAD_REQUEST
            );
        }

        String otp = UserUtils.generateOTP(6);
        this.redisService.setValue(email, otp, 60 * 60 * 5);

        try {
            this.emailService.sendOtpEmail(email, otp);
            return new ResponseEntity<>(
                    new GeneralResponse(
                            true,
                            202,
                            "OTP send to "+ email,
                            null
                    ),
                    HttpStatus.CREATED
            );

        } catch (MessagingException e) {
            return new ResponseEntity<>(
                    new GeneralResponse(
                            false,
                            500,
                            "Failed to send otp!",
                            null
                    ),
                    HttpStatus.INTERNAL_SERVER_ERROR
            );
        }
    }

    @PostMapping("/verify-otp")
    public ResponseEntity<GeneralResponse> verifyOtp(
            @RequestParam String email,
            @RequestParam String otp
    ) {

        String generatedOtp = redisService.getValue(email);
        if(generatedOtp != null && generatedOtp.equals(otp)) {
            return new ResponseEntity<>(
                    new GeneralResponse(
                            true,
                            202,
                            "OTP verified successfully!",
                            null
                    ),
                    HttpStatus.OK
            );
        } else {
            return new ResponseEntity<>(
                    new GeneralResponse(
                            false,
                            401,
                            "Failed to verify OTP!",
                            null
                    ),
                    HttpStatus.UNAUTHORIZED
            );
        }
    }



    @PostMapping("/add")
    public ResponseEntity<GeneralResponse> saveNewUser(
           @RequestBody User user
    )  {
        user.setRoleId(1L);
        this.userService.saveNewUser(user);
        return new ResponseEntity<>(
                new GeneralResponse(
                        true,
                        201,
                        "User created successfully!",
                        null
                ),
                HttpStatus.OK
        );
    }

    @GetMapping("/get-all")
    public ResponseEntity<GeneralResponse> getUserList() {
        return new ResponseEntity<>(
                new GeneralResponse(
                        true,
                        202,
                        "Data fetched successfully!",
                        this.userService.getUserList()
                ),
                HttpStatus.OK
        );
    }


}
