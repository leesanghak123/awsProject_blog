package com.cos.blog.controller.api;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import com.cos.blog.config.auth.PrincipalDetail;
import com.cos.blog.dto.ReplyReplySaveRequestDto;
import com.cos.blog.dto.ReplySaveRequestDto;
import com.cos.blog.dto.ResponseDto;
import com.cos.blog.model.Board;
import com.cos.blog.model.Reply;
import com.cos.blog.model.RoleType;
import com.cos.blog.model.User;
import com.cos.blog.service.BoardService;
import com.cos.blog.service.UserService;

import ch.qos.logback.core.model.Model;
import jakarta.servlet.http.HttpSession;
import jakarta.validation.Valid;
import reactor.core.publisher.Mono;

@RestController
public class BoardApiController {
	
	@Autowired
	private BoardService boardService;
	
	@PostMapping("/api/board")
	public ResponseDto<Integer> save(@Valid @RequestBody Board board, @AuthenticationPrincipal PrincipalDetail principal) { // username, password, email
		boardService.글쓰기(board, principal.getUser());
		return new ResponseDto<Integer>(HttpStatus.OK.value(), 1); // 자바오브젝트를 JSON으로 변환해서 리턴(Jackson), OK = 정상적이면 200, result = 정상이면 1 아니면 -1 (Userservice)
	}
	
	@DeleteMapping("/api/board/{id}")
	public ResponseDto<Integer> deleteById(@PathVariable int id){
		boardService.글삭제하기(id);
		return new ResponseDto<Integer>(HttpStatus.OK.value(), 1); // 자바오브젝트를 JSON으로 변환해서 리턴(Jackson), OK = 정상적이면 200, result = 정상이면 1 아니면 -1 (Userservice)
	}
	
	@PutMapping("/api/board/{id}")
	public ResponseDto<Integer> update(@Valid @PathVariable int id, @RequestBody Board board) {
		System.out.println("id"+id);
		System.out.println("1"+board.getTitle());
		System.out.println("2"+board.getContent());
	    boardService.글수정하기(id, board);
	    return new ResponseDto<Integer>(HttpStatus.OK.value(), 1);
	}
	
	// 데이터를 받을 때 컨트롤러에서 dto를 만들어서 받는게 좋다.
	// dto를 사용하지 않은 이유는!! 
	@PostMapping("/api/board/{boardId}/reply")
	public ResponseDto<Integer> replySave(@Valid @RequestBody ReplySaveRequestDto replySaveRequestDto) {
		boardService.댓글쓰기(replySaveRequestDto);
		return new ResponseDto<Integer>(HttpStatus.OK.value(), 1);
	}
	
	@DeleteMapping("/api/board/{boardId}/reply/{replyId}")
	public ResponseDto<Integer> replySave(@PathVariable int replyId) {
		boardService.댓글삭제(replyId);
		return new ResponseDto<Integer>(HttpStatus.OK.value(), 1);
	}
	
	@PostMapping("/api/board/{boardId}/good")
    public ResponseDto<Integer> good(@PathVariable int boardId, @AuthenticationPrincipal PrincipalDetail principal) {
        boardService.좋아요(principal.getUser().getId(), boardId);
        return new ResponseDto<Integer>(HttpStatus.OK.value(), 1);
    }
	
	@PostMapping("/api/reply/{replyId}/reply")
    public ResponseDto<Integer> replyReplySave(@RequestBody ReplyReplySaveRequestDto replyReplySaveRequestDto) {
        boardService.대댓글쓰기(replyReplySaveRequestDto);
        return new ResponseDto<>(HttpStatus.OK.value(), 1);
    }
	
	@DeleteMapping("/api/board/{boardId}/reply/{replyId}/{replyReplyId}")
	public ResponseDto<Integer> replyReplyId(@PathVariable int replyReplyId) {
		boardService.대댓글삭제(replyReplyId);
		return new ResponseDto<Integer>(HttpStatus.OK.value(), 1);
	}
}
