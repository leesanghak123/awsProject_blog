package com.cos.blog.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import com.cos.blog.model.User;

// DAO
// 자동으로 bean 등록이 된다.
// @Repository 생략 가능
public interface UserRepository extends JpaRepository<User, Integer>{
	// SELECT * FROM user WHERE username = 1?;
	// findByUsername의 Username은 (첫글자 대문자) 는 WHERE절 뒤에 나오는 부분
	Optional<User> findByUsername(String username);
}



// JPA Naming 쿼리 전략
// SELECT * FROM user WHERE username = ?(username) AND password = ?(password);
// User findByUsernameAndPassword(String username, String password);

// 위의 코드와 같은 뜻
//@Query(value="SELECT * FROM user WHERE username = ?1 AND password = ?2", nativeQuery = true)
//User login(String username, String password);