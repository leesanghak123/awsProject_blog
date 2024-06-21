package com.cos.blog.model;

import java.sql.Timestamp;
import java.util.List;

import org.hibernate.annotations.CreationTimestamp;

import com.cos.blog.dto.ReplySaveRequestDto;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
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
public class Reply {

	@Id // Primary key
	@GeneratedValue(strategy = GenerationType.IDENTITY)  // 프로젝트에서 연결된 DB의 넘버링 전략을 따라간다.
	private int id;  // auto_increment
	
	@NotBlank
	@Column(nullable = false, length = 200)
	private String content;
	
	@ManyToOne // OneToOne 은 하나의 게시글의 하나의 답변
	@JoinColumn(name = "boardId")
	private Board board;
	
	@ManyToOne() // 하나의 유저는 여러개의 답변을 작성할 수 있다
	@JoinColumn(name = "userId")
	private User user;
	
	@OneToMany(mappedBy = "reply", fetch = FetchType.EAGER, cascade = CascadeType.REMOVE)  // mappedBy 연관관계의 주인이 아니다. (난 FK가 아니에요), DB에 컬럼을 만들지 마세요, cascade = CascadeType.REMOVE 게시판 지울 시 댓글들도 다 삭제
	// Join은 필요없다(FK가 있으면 제1정규형을 위배하게 된다), reply 테이블에 boardId가 있다
	@JsonIgnoreProperties({"reply", "user"})  // replyreply오브젝트를 들고올 때 reply를 다시 호출하지 않는다 (무한참조 방지)
	private List<ReplyReply> replyReplies;  // 게시글에 여러개의 댓글이니까 List 형태로 받는다.
	
	@CreationTimestamp
	private Timestamp createDate;

	// 우클릭 -> source -> generration toSrting로 만듦
	@Override
	public String toString() {
		return "Reply [id=" + id + ", content=" + content + ", board=" + board + ", user=" + user + ", createDate="
				+ createDate + "]";
	}
	
	
}
