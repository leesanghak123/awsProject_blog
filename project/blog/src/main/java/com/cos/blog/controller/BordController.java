package com.cos.blog.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestParam;

import com.cos.blog.config.auth.PrincipalDetail;
import com.cos.blog.model.Board;
import com.cos.blog.service.BoardService;
import com.cos.blog.service.TravelPlanService;

@Controller
public class BordController {

	// 컨트롤러에서 세션을 어떻게 찾는지?
	// @AuthenticationPrincipal PrincipalDetail principal 세션 접근
	@Autowired
	private BoardService boardService;
	
	@Autowired
	private TravelPlanService travelPlanService;
	
	@GetMapping({"","/"}) // /요청을 하면
	public String  index(Model model, @PageableDefault(size=10,sort="id",direction = Sort.Direction.DESC) Pageable pageable) {// DESC는 최근 
		// /WEB-INF/views/index.jsp (yml에서 저장한 경로)
		model.addAttribute("boards", boardService.글목록(pageable));
		return "index"; // @Controller에 의해 viewResolver 작동
	}
	
	// 좋아요 버튼 동적
	@GetMapping("/board/{id}")
	public String findById(@PathVariable int id, @AuthenticationPrincipal PrincipalDetail principal, Model model) {
	    Board board = boardService.글상세보기(id);
	    boolean isGoodPressed = boardService.좋아요여부(principal.getUser().getId(), id);
	    model.addAttribute("board", board);
	    model.addAttribute("isGoodPressed", isGoodPressed);
	    return "board/detail";
	}
	
	@GetMapping("/board/{id}/updateForm")
	public String updateForm(@PathVariable int id, Model model) {
		model.addAttribute("board",boardService.글상세보기WithoutCount(id));
		return "board/updateForm";
	}
	
	@GetMapping("/board/saveForm")
	public String saveForm() {
		return "board/saveForm";
	}
}
