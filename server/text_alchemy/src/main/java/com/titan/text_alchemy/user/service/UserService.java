package com.titan.text_alchemy.user.service;

import com.titan.text_alchemy.user.model.User;
import com.titan.text_alchemy.user.repository.UserRepository;
import jakarta.validation.constraints.NotNull;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class UserService {

    @Autowired
    private UserRepository userRepository;

    public void saveNewUser(@NotNull User user) {
        this.userRepository.save(user);
    }

    public List<User> getUserList() {
        return this.userRepository.findAll();
    }

    public User getUserByEmail(@NotNull String email) {

        return this.userRepository.findById(email).orElse(null);

    }
}
