 package com.cos.blog.config;

import org.springframework.boot.autoconfigure.session.RedisSessionProperties.ConfigureAction;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;

// 1. 어노테이션 제거
@Configuration
public class SecurityConfig{ // 2. extends 제거

	// 3. principalDetailService 제거

	// 4. AuthenticationManager 메서드 생성 (스프링 시큐리티 인증을 처리)
	// PrincipalDetailService와 연결되는 인증필터체인
	@Bean
	public AuthenticationManager authenticationManager(AuthenticationConfiguration authenticationConfiguration) throws Exception {
		return authenticationConfiguration.getAuthenticationManager();
	}
	
	@Bean // IoC가 되요!!
	public BCryptPasswordEncoder encodePWD() {
		return new BCryptPasswordEncoder();
	}
	
	// 5. 기본 패스워드 체크가 BCryptPasswordEncoder 여서 설정 필요 없음.
	
	// 6. 최신 버전(2.7)으로 시큐리티 필터 변경
	@Bean
	SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
		// 1. csrf 비활성화
		http.csrf(c -> c.disable());

		// 2. 인증 주소 설정 (WEB-INF/** 추가해줘야 함. 아니면 인증이 필요한 주소로 무한 리다이렉션 일어남)
		http.authorizeHttpRequests(authorize -> authorize
				.requestMatchers("/WEB-INF/**","/", "/auth/**", "/js/**", "/css/**", "/image/**", "/dummy/**").permitAll()
				.anyRequest().authenticated());
		
		// 3. 로그인 처리 프로세스 설정 (허용안되는 곳으로 갈 시 URL설정)
		http.formLogin(f -> f.loginPage("/auth/loginForm")
				.loginProcessingUrl("/auth/loginProc")
				.defaultSuccessUrl("/") // 로그인 성공시 기본 페이지
		);

		return http.build();
	}
}