package com.titan.text_alchemy.user.model;


import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Entity
@AllArgsConstructor
@NoArgsConstructor
@Setter
@Getter
public class User {

    @Id
    private Long id;
    private String firstName;
    private String lastName;
    private String username;
    private String email;
    private boolean active;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
