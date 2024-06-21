package com.cos.blog.controller.api;

import java.util.HashMap;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.jpa.repository.Query;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import com.cos.blog.config.auth.PrincipalDetail;
import com.cos.blog.dto.ResponseDto;
import com.cos.blog.model.Gpt;
import com.cos.blog.model.User;
import com.cos.blog.service.TravelPlanService;
import com.cos.blog.service.UserService;

import jakarta.validation.Valid;
import reactor.core.publisher.Mono;

@RestController
public class UserApiController {

	@Autowired
	private TravelPlanService travelPlanService;

	@Autowired
	private UserService userService;

	@Autowired
	private AuthenticationManager authenticationManager;

	// 회원 가입 처리
	@PostMapping("/auth/joinProc")
	public ResponseDto<Integer> save(@Valid @RequestBody User user) {
		System.out.println("UserApiController: save 호출됨");
		userService.회원가입(user);
		return new ResponseDto<>(HttpStatus.OK.value(), 1);
	}

	// 회원 정보 수정
	@PutMapping("/user")
	public ResponseDto<Integer> update(@Valid @RequestBody User user) {
		userService.회원수정(user);
		Authentication authentication = authenticationManager.authenticate(new UsernamePasswordAuthenticationToken(user.getUsername(), user.getPassword()));
		SecurityContextHolder.getContext().setAuthentication(authentication);
		return new ResponseDto<>(HttpStatus.OK.value(), 1);
	}

	// 여행 계획 생성 요청 처리
//	@PostMapping("/api/user/travel/plan")
//	public Mono<ResponseEntity<Map<String, String>>> createTravelPlan(@Valid @RequestBody Map<String, String> requestData) {
//		String stage = requestData.get("stage");
//		String kind = requestData.get("kind");
//		String content = "장소는 " + stage + " 곳에서 " + "음식의 종류는 " + kind + "인데 이걸 기반으로 먹을 곳을 추천해줘";
//
//		return travelPlanService.여행계획(content)
//				.map(plan -> {
//					Map<String, String> response = new HashMap<>();
//					response.put("plan", plan);
//					return new ResponseEntity<>(response, HttpStatus.OK);
//				})
//				.defaultIfEmpty(new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR));
//	}

	// Python FastAPI에 쿼리 요청 보내기
	@PostMapping("/api/user/travel/plan")
	public ResponseDto<Map<String, String>> write(@RequestBody Gpt gpt) {
	    System.out.println("Received Gpt data: " + gpt); // Debugging line
	    String plan = travelPlanService.여행계획(gpt).block(); // Blocking to wait for the result
	    Map<String, String> response = new HashMap<>();
	    response.put("plan", plan);
	    return new ResponseDto<>(HttpStatus.OK.value(), response);
	}

	// 여행 계획 저장 처리
	@PostMapping("/api/user/travel/save")
	public ResponseDto<Integer> planSave(@RequestBody Gpt gpt, @AuthenticationPrincipal PrincipalDetail principal) {
		travelPlanService.계획저장(gpt, principal.getUser());
		return new ResponseDto<>(HttpStatus.OK.value(), 1);
	}

	// 여행 계획 수정 처리
	@PutMapping("/api/user/planUpdate/{id}")
	public ResponseDto<Integer> planUpdate(@Valid @PathVariable int id, @RequestBody Gpt gpt) {
		travelPlanService.글수정하기(id, gpt);
		return new ResponseDto<>(HttpStatus.OK.value(), 1);
	}

	// 여행 계획 삭제 처리
	@DeleteMapping("/api/user/{id}")
	public ResponseDto<Integer> planDelete(@PathVariable int id) {
		travelPlanService.글삭제하기(id);
		return new ResponseDto<>(HttpStatus.OK.value(), 1);
	}
}
