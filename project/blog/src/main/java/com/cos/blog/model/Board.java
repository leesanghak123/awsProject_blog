package com.cos.blog.model;

import java.sql.Timestamp;
import java.util.List;

import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.DialectOverride.ColumnDefault;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.Lob;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToMany;
import jakarta.persistence.OrderBy;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder // 빌더 패턴
@Entity
public class Board {
	
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY) // auto_increment
	private int id;
	
	@NotBlank
	@Column(nullable = false, length = 100)
	private String title;
	
	//@Lob // 대용량 데이터 : TINYTEXT으로 등록되어 에러남.
	@NotBlank
	@Column(columnDefinition = "longtext")
	private String content; // 섬머노트 라이브러리 <html>태그가 섞여서 디자인 됨.

	private int count; // 조회수
	
	private int good; // 좋아요
	
	@ManyToOne(fetch = FetchType.EAGER)  // Many = Board, One = User, 한 명의 유저는 여러개의 개시글 작성 가능
	@JoinColumn(name="userId")
	private User user; // DB는 오브젝트를 저장할 수 없다. FK, 자바는 오브젝트를 저장할 수 있다.
	
	@OneToMany(mappedBy = "board", fetch = FetchType.EAGER, cascade = CascadeType.REMOVE)  // mappedBy 연관관계의 주인이 아니다. (난 FK가 아니에요), DB에 컬럼을 만들지 마세요, cascade = CascadeType.REMOVE 게시판 지울 시 댓글들도 다 삭제
	// Join은 필요없다(FK가 있으면 제1정규형을 위배하게 된다), reply 테이블에 boardId가 있다
	@JsonIgnoreProperties({"board", "user"})  // reply오브젝트를 들고올 때 board를 다시 호출하지 않는다 (무한참조 방지)
	@OrderBy("id desc") // 댓글 내림차순 정렬
	private List<Reply> replys;  // 게시글에 여러개의 댓글이니까 List 형태로 받는다.
	
	@CreationTimestamp
	private Timestamp createDate;  // 현재 시간
}
