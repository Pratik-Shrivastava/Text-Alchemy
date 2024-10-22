package com.titan.text_alchemy.common.service;

import lombok.AllArgsConstructor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Duration;

@Service
public class RedisService {

    @Autowired
    private StringRedisTemplate redisTemplate;

    public void setValue(String key, String value) {
        redisTemplate.opsForValue().set(key, value);
    }

    public void setValue(String key, String value, long timeoutInSeconds) {
        redisTemplate.opsForValue().set(key, value, Duration.ofSeconds(timeoutInSeconds));
    }

    public String getValue(String key) {
        return redisTemplate.opsForValue().get(key);
    }

}
