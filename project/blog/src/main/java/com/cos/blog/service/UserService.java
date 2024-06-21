package com.cos.blog.service;

import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.reactive.function.client.WebClient;

import com.cos.blog.model.Gpt;
import com.cos.blog.model.RoleType;
import com.cos.blog.model.User;
import com.cos.blog.repository.UserRepository;

import jakarta.validation.Valid;
import reactor.core.publisher.Mono;




// 스프링이 컴포넌트 스캔을 통해서 Bean에 등록해줌.(IoC를 해준다.(메모리에 대신 띄어준다))
@Service
public class UserService {

	@Autowired
	public UserRepository userRepository;
	
	@Autowired // DI 주입
	private BCryptPasswordEncoder encoder;
	
	@Transactional(readOnly = true)
	public User 회원찾기(String username) {
		// user 이름이 없는 경우 빈객체를 리턴 (Get으로 null을 리턴하려 했는데 오류)
		User user = userRepository.findByUsername(username).orElseGet(()->{
			return new User();
		});
		return user;
	}
	
	@Transactional // 회원가입 전체가 하나의 트랜잭션으로 묶임(전체 성공 시 commit)
	public void 회원가입(User user) {
		String rawPassword = user.getPassword(); // 1234 원문
		String encPassword = encoder.encode(rawPassword); // 해쉬
		user.setPassword(encPassword);
		user.setRole(RoleType.User); // role도 입력해줘야 하기 때문에 설정
		userRepository.save(user);
	}
	
	@Transactional
	public void 회원수정(User user) {
		// 수정 시에는 영속성 컨텍스트 User 오브젝트를 영속화시키고, 영속화된 User 오브젝트를 수정
		// select를 해서 User오브젝트를 DB로 부터 가져오는 이유는 영속화를 하기 위해서!
		// 영속화된 오브젝트를 변경하면 자동으로 DB에 update 문을 날려주거든요
		User persistance = userRepository.findById(user.getId()).orElseThrow(()->{
			return new IllegalArgumentException("회원 찾기 실패");
		});
		
		// Validate 체크 (카카오 로그인인 사용자는 패스워드 수정을 못함)_서버쪽에서 post공격 대비
		// 일반 사용자만 수정 가능(oauth값이 필드에 없으면)
		if(persistance.getOauth() == null || persistance.getOauth().equals("")) {
			String rawPassword = user.getPassword();
			String encPassword = encoder.encode(rawPassword);
			persistance.setPassword(encPassword);
			persistance.setEmail(user.getEmail());
		}
		// 회원수정 함수 종료시 = 서비스 종료 = 트랜잭션 종료 = commit 이 자동으로 됩니다
		// 영속화된 persistance 객체의 변화가 감지되면 더티체킹이 되어 update문을 날려줌
    }
}
