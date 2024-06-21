// 수정확인
package com.cos.blog.config.auth;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;

import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import com.cos.blog.model.User;

import lombok.Getter;

// 스프링 시큐리티가 로그인 요청을 가로채서 로그인을 진행하고 완료되면
// UserDetails 타입의 오브젝트를 스프링 시큐리티의 고유한 세션저장소에 저장을 해준다
@Getter // user정보를 들고와서 게시판에 적용하기 위함
public class PrincipalDetail implements UserDetails{
	// 콤포지션 (extends가 아니라 품고 있는 것)
	private User user;  // user.java에 있는 유저(model)

	public PrincipalDetail(User user) {
		this.user = user;
	}
	
	// alt + shift + s 로 override 한 것
	@Override
	public String getPassword() {
		return user.getPassword();
	}

	@Override
	public String getUsername() {
		return user.getUsername();
	}

	// 계정이 만료되지 않았는지 리턴한다 (true: 만료안됨)
	@Override
	public boolean isAccountNonExpired() {
		return true;
	}

	// 계정이 잠겨있는지 리턴한다 (true: 잠기지않음)
	@Override
	public boolean isAccountNonLocked() {
		return true;
	}

	// 비밀번호가 만료되지 않았는지 (true: 만료안됨)
	@Override
	public boolean isCredentialsNonExpired() {
		return true;
	}

	// 계정이 활성화(사용가능)인지 리턴한다 (true: 활성화)
	@Override
	public boolean isEnabled() {
		return true;
	}
	
	// 계정의 권한목록을 리턴한다 (권한이 여러개 있을 수 있어서 for문을 돌려야하는데 우리는 한개만)
	@Override
	public Collection<? extends GrantedAuthority> getAuthorities() {
		
		Collection<GrantedAuthority> collectors = new ArrayList<>();
		// 스프링에서 role을 받을 때 규칙 아니면 인식을 못함, ROLE_
		collectors.add(()->{return "ROLE_"+user.getRole();});
		
		return collectors;
	}
}
