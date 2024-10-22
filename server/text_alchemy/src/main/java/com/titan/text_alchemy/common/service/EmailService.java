package com.titan.text_alchemy.common.service;


import lombok.AllArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Service;

import jakarta.mail.MessagingException;
import jakarta.mail.internet.MimeMessage;
import jakarta.mail.util.ByteArrayDataSource;
import java.io.ByteArrayInputStream;
import java.io.IOException;

@Service
@AllArgsConstructor
public class EmailService {

    @Autowired
    private final JavaMailSender mailSender;

    public void sendOtpEmail(String to, String otp) throws MessagingException {
        String subject = "Your OTP Code";
        String htmlBody = "<html>" +
                "<body>" +
                "<h2>Your One-Time Password (OTP)</h2>" +
                "<p>Dear User,</p>" +
                "<p>Your OTP for verification is: <strong>" + otp + "</strong></p>" +
                "<p>Please use this OTP to login into your Text Alchemy account.</p>" +
                "<p>Thank you!</p>" +
                "<p>Regards,<br>Team Titan</p>" +
                "</body>" +
                "</html>";

        sendHtmlEmail(to, subject, htmlBody);
    }

    public void sendHtmlEmail(String to, String subject, String htmlBody) throws MessagingException {
        MimeMessage message = mailSender.createMimeMessage();
        MimeMessageHelper helper = new MimeMessageHelper(message, true, "UTF-8");
        helper.setSubject(subject);
        helper.setTo(to);
        helper.setText(htmlBody, true); // true indicates HTML

        mailSender.send(message);
    }

    public void sendHtmlEmail(String to, String subject, String htmlBody, ByteArrayInputStream attachment) throws MessagingException, IOException {
        MimeMessage message = mailSender.createMimeMessage();
        MimeMessageHelper helper = new MimeMessageHelper(message, true);
        helper.setSubject(subject);
        helper.setTo(to);
        helper.setText(htmlBody, true); // true indicates HTML
        ByteArrayDataSource dataSource = new ByteArrayDataSource(attachment, "application/pdf");
        helper.addAttachment("interaction.pdf", dataSource);

        mailSender.send(message);
    }

    public void sendHtmlEmail(String to, String[] bcc, String subject, String htmlBody, ByteArrayInputStream attachment) throws MessagingException, IOException {
        MimeMessage message = mailSender.createMimeMessage();
        MimeMessageHelper helper = new MimeMessageHelper(message, true);
        helper.setSubject(subject);
        helper.setTo(to);
        helper.setBcc(bcc);
        helper.setText(htmlBody, true); // true indicates HTML
        ByteArrayDataSource dataSource = new ByteArrayDataSource(attachment, "application/pdf");
        helper.addAttachment("interaction.pdf", dataSource);

        mailSender.send(message);
    }

}
