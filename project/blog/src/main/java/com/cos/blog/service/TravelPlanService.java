package com.cos.blog.service;

import com.cos.blog.model.Gpt;
import com.cos.blog.model.User;
import com.cos.blog.repository.GptRepository;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.HashMap;
import java.util.Map;

@Service
public class TravelPlanService {

    private final WebClient webClient;
    private final GptRepository gptRepository;

    @Autowired
    public TravelPlanService(WebClient.Builder webClientBuilder, GptRepository gptRepository) {
        this.webClient = webClientBuilder.baseUrl("http://localhost:8001").build(); // Python FastAPI 서버 URL
        this.gptRepository = gptRepository;
    }

    public Mono<String> 여행계획(Gpt gpt) {
        Map<String, String> requestBody = new HashMap<>();
        requestBody.put("question", gpt.getStage()); // "question" 필드에 stage 값을 넣음

        return webClient.post()
                .uri("/query")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(requestBody)
                .retrieve()
                .bodyToMono(String.class)
                .flatMap(responseBody -> {
                    // JSON 응답에서 "answer" 필드 추출
                    try {
                        ObjectMapper mapper = new ObjectMapper();
                        JsonNode jsonNode = mapper.readTree(responseBody);
                        String answer = jsonNode.get("answer").asText();
                        return Mono.just(answer);
                    } catch (Exception e) {
                        return Mono.error(e);
                    }
                })
                .doOnError(e -> System.err.println("WebClient error: " + e.getMessage())); // 오류 발생 시 디버깅 메시지 출력
    }


//	// OpenAI API를 사용하여 여행 계획 생성 요청을 보내고 결과를 Mono<String>으로 반환
//	public Mono<String> 여행계획(String content) {
//		return webClient.post().uri("/chat/completions")
//				.bodyValue(Map.of("model", "gpt-3.5-turbo-16k", "messages", List.of(
//						Map.of("role", "system", "content", "너는 food 가이드 역할이야"),
//						Map.of("role", "user", "content", content))))
//				.retrieve().bodyToMono(Map.class)
//				.flatMap(response -> {
//					List<Map<String, Object>> choices = (List<Map<String, Object>>) response.get("choices");
//					if (choices != null && !choices.isEmpty()) {
//						Map<String, Object> message = (Map<String, Object>) choices.get(0).get("message");
//						String resultContent = (String) message.get("content");
//						return Mono.just(resultContent);
//					} else {
//						return Mono.error(new RuntimeException("Invalid response format"));
//					}
//				});
//	}

	// 여행 계획 저장 메서드
	@Transactional
	public void 계획저장(Gpt gpt, User user) {
		gpt.setUser(user);
		gpt.setStage(gpt.getStage());
		//gpt.setEnd(gpt.getEnd());
		//gpt.setDays(gpt.getDays());
		gpt.setResult(gpt.getResult());
		gptRepository.save(gpt);
	}

	// 사용자별로 저장된 여행 계획 목록 조회
	@Transactional(readOnly = true)
	public Page<Gpt> 사용자별글목록(String username, Pageable pageable) {
		return gptRepository.findByUserUsername(username, pageable);
	}

	// 특정 여행 계획 상세 조회
	@Transactional(readOnly = true)
	public Gpt 글상세보기(int id) {
		return gptRepository.findById(id).orElseThrow(() -> new IllegalArgumentException("글 상세보기 실패: 아이디를 찾을 수 없습니다."));
	}

	// 여행 계획 수정
	@Transactional
	public void 글수정하기(int id, Gpt requestGpt) {
		Gpt gpt = gptRepository.findById(id).orElseThrow(() -> new IllegalArgumentException("글 찾기 실패: 아이디를 찾을 수 없습니다."));
		gpt.setResult(requestGpt.getResult());
	}

	// 여행 계획 삭제
	@Transactional
	public void 글삭제하기(int id) {
		gptRepository.deleteById(id);
	}
}