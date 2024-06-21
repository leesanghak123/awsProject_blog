package com.cos.blog.service;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.cos.blog.dto.ReplyReplySaveRequestDto;
import com.cos.blog.dto.ReplySaveRequestDto;
import com.cos.blog.model.Board;
import com.cos.blog.model.Good;
import com.cos.blog.model.Reply;
import com.cos.blog.model.ReplyReply;
import com.cos.blog.model.RoleType;
import com.cos.blog.model.User;
import com.cos.blog.repository.BoardRepository;
import com.cos.blog.repository.GoodRepository;
import com.cos.blog.repository.ReplyReplyRepository;
import com.cos.blog.repository.ReplyRepository;
import com.cos.blog.repository.UserRepository;

import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor // @Autowired 대신 final을 붙이는데 final은 변수 초기화를 해줘야 해서 붙이는 어노테이션
public class BoardService {

	private final BoardRepository boardRepository;
	private final ReplyRepository replyRepository;
	private final GoodRepository goodRepository;
	private final UserRepository userRepository;
	private final ReplyReplyRepository replyReplyRepository;
	
//public BoardService(BoardRepository bRepo, ReplyRepository rRepo) {
//	this.boardRepository = bRepo;
//	this.replyRepository = rRepo;
//}
	
	@Transactional
	public void 글쓰기(Board board, User user) { // title, content
		board.setCount(0);
		board.setGood(0);
		board.setUser(user);
		boardRepository.save(board);
	}
	
	@Transactional(readOnly = true)  // select만 하는 거니까 readOnly = true
	public Page<Board> 글목록(Pageable pageable){
		return boardRepository.findAll(pageable);
	}
	
	@Transactional
	public Board 글상세보기(int id) {
		// 조회수 증가
        boardRepository.incrementCount(id);
		return boardRepository.findById(id)
				.orElseThrow(()->{
					return new IllegalArgumentException("글 상세보기 실패: 아이디를 찾을 수 없습니다.");
				});
	}
	
	@Transactional(readOnly = true)
	public Board 글상세보기WithoutCount(int id) {
		return boardRepository.findById(id)
				.orElseThrow(()->{
					return new IllegalArgumentException("글 상세보기 실패: 아이디를 찾을 수 없습니다.");
				});
	}
	
	@Transactional
	public void 글삭제하기(int id) {
		boardRepository.deleteById(id);
	}
	
	@Transactional
	public void 글수정하기(int id, Board requestBoard) {
		Board board = boardRepository.findById(id)
				.orElseThrow(()->{
					return new IllegalArgumentException("글 찾기 실패: 아이디를 찾을 수 없습니다.");
				});	// 영속화 완료
		board.setTitle(requestBoard.getTitle());
		board.setContent(requestBoard.getContent());
		// 해당 함수로 종료시(Service가 종료될 때) 트랜잭션이 종료됩니다. 이때 더티채킹 - 자동 업데이트. DB flush(commit이 된다는 뜻)
	}
	
	@Transactional
	public void 댓글쓰기(ReplySaveRequestDto replySaveRequestDto) {
		int result = replyRepository.mSave(replySaveRequestDto.getUserId(), replySaveRequestDto.getBoardId(), replySaveRequestDto.getContent());
		System.out.println(result);
	}
	
	@Transactional
	public void 댓글삭제(int replyId) {
		replyRepository.deleteById(replyId);
	}
	
	@Transactional
    public void 좋아요(int userId, int boardId) {
        User user = userRepository.findById(userId).orElseThrow(() -> new IllegalArgumentException("사용자를 찾을 수 없습니다."));
        Board board = boardRepository.findById(boardId).orElseThrow(() -> new IllegalArgumentException("게시물을 찾을 수 없습니다."));
        
        Good existingGood = goodRepository.findByUserAndBoard(user, board);
        if (existingGood != null) {
        	// 좋아요 취소
            goodRepository.delete(existingGood);
            board.setGood(board.getGood() - 1);
        } else {
        	Good good = new Good();
            good.setUser(user);
            good.setBoard(board);
            goodRepository.save(good);

            // 좋아요 수 증가
            board.setGood(board.getGood() + 1);
        }
        boardRepository.save(board);
    }
	
	public boolean 좋아요여부(int userId, int boardId) {
        User user = userRepository.findById(userId).orElseThrow(() -> new IllegalArgumentException("사용자를 찾을 수 없습니다."));
        Board board = boardRepository.findById(boardId).orElseThrow(() -> new IllegalArgumentException("게시물을 찾을 수 없습니다."));
        Good existingGood = goodRepository.findByUserAndBoard(user, board);
        return existingGood != null;
    }
	
	@Transactional
    public void 대댓글쓰기(ReplyReplySaveRequestDto replyReplySaveRequestDto) {
        ReplyReply replyReply = new ReplyReply();
        replyReply.setUser(userRepository.findById(replyReplySaveRequestDto.getUserId())
                .orElseThrow(() -> new IllegalArgumentException("사용자를 찾을 수 없습니다.")));
        replyReply.setReply(replyRepository.findById(replyReplySaveRequestDto.getReplyId())
                .orElseThrow(() -> new IllegalArgumentException("댓글을 찾을 수 없습니다.")));
        replyReply.setContent(replyReplySaveRequestDto.getContent());

        replyReplyRepository.save(replyReply);
    }
	
	@Transactional
	public void 대댓글삭제(int replyReplyId) {
		replyReplyRepository.deleteById(replyReplyId);
	}

}
