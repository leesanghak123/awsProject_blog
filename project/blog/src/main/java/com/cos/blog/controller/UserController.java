package com.cos.blog.controller;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.context.HttpSessionSecurityContextRepository;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.client.RestTemplate;

import com.cos.blog.config.auth.PrincipalDetail;
import com.cos.blog.model.Gpt;
import com.cos.blog.model.KakaoProfile;
import com.cos.blog.model.OAuthToken;
import com.cos.blog.model.User;
import com.cos.blog.repository.GptRepository;
import com.cos.blog.service.TravelPlanService;
import com.cos.blog.service.UserService;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import jakarta.servlet.http.HttpSession;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.web.PageableDefault;

// 인증이 안된 사용자들이 출입할 수 있는 경로를 /auth/** 허용 (인증이 필요 없는 부분_회원가입하는 부분, 로그인 하는 부분)
// 그냥 주소가 / 이면 index.jsp 허용
// static 이하에 있는 /js/**, /css/**, /image/**

@Controller
public class UserController {
	
	// yml의 cos.key 사용
	// import org.springframework.beans.factory.annotation.Value; 이걸 직접 import 해줌
	@Value("${cos.key}")
	private String cosKey;
	
	@Autowired
	private AuthenticationManager authenticationManager;
	
	@Autowired
	private UserService userService;
	
	@Autowired
	private TravelPlanService travelPlanService;
	
	@Autowired
	private HttpSession session;
	
	@GetMapping("/auth/joinForm")
	public String joinForm() {
		return "user/joinForm";
	}
	
	@GetMapping("/auth/loginForm")
	public String loginForm() {
		return "user/loginForm";
	}
	
	@GetMapping("/auth/kakao/callback")
	public String kakaoCallback(String code) { // @ResponseBody 는 Data를 리턴해주는 컨트롤러 함수(지움) -> 아래 return "redirect:/"; 를 뷰리졸버를 통해 /로 이동하기 위해
		
		// POST방식으로 key=value 데이터를 요청 (카카오쪽으로)
		// a태그 방식의 요청은 무조건 GET방식
		// Retrofit2, OkHttp, Rest Template 등의 라이브러리를 사용할 수 있다 (http요청을 편하게 할 수 있다)
		RestTemplate rt = new RestTemplate();
		
		// HttpHeader 오브젝트 생성
		HttpHeaders headers = new HttpHeaders();
		headers.add("Content-type", "application/x-www-form-urlencoded;charset=utf-8"); // Content-type을 담는다는 의미는 내가 지금 전송할 body 데이터가 key-value형태라는 것을 알려준다
		
		// HttpBody 오브젝트 생성 (원래는 변수화 시켜서 사용)
		MultiValueMap<String, String> params = new LinkedMultiValueMap<>();
		params.add("grant_type", "authorization_code");
		params.add("client_id", "6ece1eddc88f259fef887cb9732ed10a");
		params.add("redirect_uri", "http://localhost:8000/auth/kakao/callback");
		params.add("code", code);
		
		// HttpHeader와 httpBody를 하나의 오브젝트(Entity)로 만들어주기
		HttpEntity<MultiValueMap<String, String>> kakaoTokenRequest =
				new HttpEntity<>(params, headers);
		
		// 실제 Http 요청 (주소, 요청매서드, header+body값, 응답을 받을 타입(string))
		ResponseEntity<String> response = rt.exchange(
				"https://kauth.kakao.com/oauth/token",
				HttpMethod.POST,
				kakaoTokenRequest,
				String.class
				);
		
		// json 데이터를 오브젝트에 담는 라이브러리 = Gson, Json Simple, ObjectMapper
		ObjectMapper objectMapper = new ObjectMapper();
		OAuthToken oauthToken = null;
		try {
			oauthToken = objectMapper.readValue(response.getBody(), OAuthToken.class);
		} catch (JsonMappingException e) {
			e.printStackTrace();
		} catch (JsonProcessingException e) {
			e.printStackTrace();
		}
		
		System.out.println("카카오 엑세스 토큰: "+oauthToken.getAccess_token());
		
		RestTemplate rt2 = new RestTemplate();
		
		// HttpHeader 오브젝트 생성
		HttpHeaders headers2 = new HttpHeaders();
		headers2.add("Authorization", "Bearer "+oauthToken.getAccess_token()); // Bearer 뒤에 한 칸 띄우기
		headers2.add("Content-type", "application/x-www-form-urlencoded;charset=utf-8"); // Content-type을 담는다는 의미는 내가 지금 전송할 body 데이터가 key-value형태라는 것을 알려준다
		
		// HttpHeader만 넘기기 (body는 필요없어서)
		HttpEntity<MultiValueMap<String, String>> kakaoProfileRequest2 =
				new HttpEntity<>(headers2);
		
		// 실제 Http 요청 (주소, 요청매서드, header+body값, 응답을 받을 타입(string))
		ResponseEntity<String> response2 = rt2.exchange(
				"https://kapi.kakao.com/v2/user/me",
				HttpMethod.POST,
				kakaoProfileRequest2,
				String.class
				);
		
		ObjectMapper objectMapper2 = new ObjectMapper();
		KakaoProfile kakaoProfile = null;
		try {
			kakaoProfile = objectMapper2.readValue(response2.getBody(), KakaoProfile.class);
		} catch (JsonMappingException e) {
			e.printStackTrace();
		} catch (JsonProcessingException e) {
			e.printStackTrace();
		}
		
		// 카카오에서 받은 값으로 회원구성하기
		// User 오브젝트에서 이메일과 아이디만 있으면 된다(role는 user, 번호는 순서대로, password는 물어볼 필요가 없으니까)
		// User 오브젝트 : username, password, email 을 넣어서 카카오에서 받은 값과 통합 시켜줘야한다
		System.out.println("카카오 아이디(번호) : "+kakaoProfile.getId());
		System.out.println("카카오 이메일 : "+kakaoProfile.getKakao_account().getEmail());
		
		// 유저네임 중복 방지를 위해서 싹 다 붙이기
		System.out.println("블로그서버 유저네임 : "+kakaoProfile.getKakao_account().getEmail()+"_"+kakaoProfile.getId());
		System.out.println("블로그서버 이메일 : "+kakaoProfile.getKakao_account().getEmail());
		// UUID란 중복되지 않는 어떤 특정 값을 만들어내는 알고리즘 (그래서 안씀)
		System.out.println("블로그서버 패스워드 : "+cosKey);
		
		User kakaoUser = User.builder()
				.username(kakaoProfile.getKakao_account().getEmail()+"_"+kakaoProfile.getId())
				.password("cosKey")
				.email(kakaoProfile.getKakao_account().getEmail())
				.oauth("kakao") // kakao를 넣어서 kakao로 로그인한 사용자 구분
				.build();
		
		// 비 가입자면 회원가입하고 로그인 처리, 가입자면 바로 로그인 처리
		// 가입자 혹은 비가입자 체크 처리
		User originUser = userService.회원찾기(kakaoUser.getUsername());
		
		if(originUser.getUsername() == null) {
			System.out.println("기존 회원이 아니기에 자동 회원가입을 진행합니다.");
			userService.회원가입(kakaoUser);
		}
		
		System.out.println("자동 로그인을 진행합니다.");
		// 로그인 처리(api 컨트롤러에 있는거 들고옴)
		// (1. UsernamePasswordAuthenticationToken 토큰 생성시 이제 정확한 정보를 추가해줘야 한다 UserDetails, password, role)
		// (2. SecurityContext에 Authentication 객체를 추가하면 예전에는 세션이 만들어졌다)
		// (3. 이제는 보안때문에, 해당 컨텍스트를 세션에 직접 주입해줘야 한다 HttpSessionSecurityContextRepository.SPRING_SECURITY_CONTEXT_KEY)
		PrincipalDetail principalDetail = new PrincipalDetail(kakaoUser);
		Authentication authentication = authenticationManager.authenticate(new UsernamePasswordAuthenticationToken(principalDetail, principalDetail.getPassword(), principalDetail.getAuthorities()));
		SecurityContext securityContext = SecurityContextHolder.getContext();
		securityContext.setAuthentication(authentication);
		session.setAttribute(HttpSessionSecurityContextRepository.
				SPRING_SECURITY_CONTEXT_KEY, securityContext);
		return "redirect:/";
	}
	
	@GetMapping("/user/updateForm")
	public String updateForm(@AuthenticationPrincipal PrincipalDetail principal) {
		return "user/updateForm";
	}
	
	@GetMapping("/user/aiPlanForm")
    public String aiPlanForm() {
        return "user/aiPlanForm";
    }
	
	@GetMapping("/user/planStorageForm")
    public String planStorageForm(@AuthenticationPrincipal PrincipalDetail principal, Model model, @PageableDefault(size=10,sort="id",direction = Sort.Direction.DESC) Pageable pageable) {
		 String username = principal.getUsername();
		model.addAttribute("gpts", travelPlanService.사용자별글목록(username, pageable));
		return "user/planStorageForm";
    }
	
	@GetMapping("/user/planDetail/{id}")
    public String planDetail(@PathVariable int id, Model model) {
		model.addAttribute("gpt", travelPlanService.글상세보기(id));
        return "user/planDetail";
    }
	
	@GetMapping("/user/planUpdate/{id}")
    public String planUpdate(@PathVariable int id, Model model) {
		model.addAttribute("gpt", travelPlanService.글상세보기(id));
        return "user/planUpdate";
    }
}
