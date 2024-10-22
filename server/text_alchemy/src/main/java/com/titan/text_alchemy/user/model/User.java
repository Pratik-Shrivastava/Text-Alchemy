package com.titan.text_alchemy.user.model;


import jakarta.persistence.*;
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
@Table(name = "users")
public class User {

    @Id
    private String email;
    private Long roleId;
    private String firstName;
    private String lastName;

    private boolean active;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
