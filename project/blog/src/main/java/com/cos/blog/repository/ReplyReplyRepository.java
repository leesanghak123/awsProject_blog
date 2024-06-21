package com.cos.blog.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;

import com.cos.blog.model.Reply;
import com.cos.blog.model.ReplyReply;

public interface ReplyReplyRepository extends JpaRepository<ReplyReply, Integer>{
	
}