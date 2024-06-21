 package com.cos.blog.model;

import java.sql.Timestamp;

import org.hibernate.annotations.ColumnDefault;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.DynamicInsert;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

// ORM -> Java(다른언어) Object -> 테이블로 매핑해주는 기술
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder // 빌더 패턴
@Entity  // User 클래스가 MySQL에 테이블이 생성된다.
// @DynamicInsert 는 insert시 null인 필드를 제외시켜준다.
public class User {
	
	@Id // Primary key
	@GeneratedValue(strategy = GenerationType.IDENTITY)  // 프로젝트에서 연결된 DB의 넘버링 전략을 따라간다.
	private int id;  // auto_increment
	
	@NotBlank
	@Column(nullable = false, length = 100, unique = true)
	private String username;  // 아이디
	
	@NotBlank
	@Column(nullable = false, length = 100)  // 100으로 넣는 이유는 해쉬 (비밀번호 암호화)
	private String password;
	
	@NotBlank
	@Email
	@Column(nullable = false, length = 50)
	private String email;
	
//	@ColumnDefault("'user'")  // 더블 안에 싱글
	// DB는 RoleType라는 게 없다.
	@Enumerated(EnumType.STRING)
	private RoleType role;  // Enum을 쓰는게 좋다.(String 대신 도메인을 써서 role에 들어가는 오타를 막는다)  // ADMIN, USER
	
	private String oauth; // kakao로 로그인 한 놈은 이걸 넣어서 회원정보 수정을 못하게 (google, kakao 등)
	
	@CreationTimestamp // 시간이 자동 입력
	private Timestamp createDate;
}
