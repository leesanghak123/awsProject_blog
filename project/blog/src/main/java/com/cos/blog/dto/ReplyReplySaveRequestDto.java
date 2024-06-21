package com.cos.blog.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

// 원래는 dto를 사용하는 것이 좋다_하지만 작은 프로젝트기 때문에
@Data
@AllArgsConstructor
@NoArgsConstructor
public class ReplyReplySaveRequestDto {
	private int userId;
	private int replyId;
	private String content;
}
